
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import collections
import os
import time
from tempfile import TemporaryDirectory


# Task1 — Fine-tuning with imbalanced classes.
#
# This script reuses the coauthor pipeline (data transforms, train_model,
# StepLR) from limited_data.py / gradual_unfreezing.py, and layers the
# imbalance experiment on top so the training regime is identical across
# strategies. The four strategies compared:
#   - baseline                    : plain CE on the imbalanced train set
#   - weighted_ce                 : nn.CrossEntropyLoss(weight = inv-freq)
#   - oversampling                : WeightedRandomSampler with inv-freq
#   - weighted_ce + oversampling  : both at once
#
# Model regime (per the report): a *linear probe* — ResNet-34 with the entire
# backbone frozen, so it emits a fixed 512-d feature vector per image, and only
# the final 512->37 linear head is trained. Optimizer is plain Adam (lr 1e-3)
# with an explicit L1 penalty on the trainable parameters (L1_LAMBDA). This is
# the regime the confusion analysis interprets: because the features are frozen,
# training can only re-aim the linear boundary, so residual confusions between
# visually similar breeds are feature-overlap errors no reweighting can fix.
#
# Generalization stack (shared by all four strategies, so the comparison is
# apples-to-apples instead of three flavors of overfit):
#   - data augmentation on train (RandomResizedCrop + flip + rotation)
#   - plain Adam (lr 1e-3) + L1 penalty on the trainable head
#   - StepLR scheduler  (drop LR by 10x every 7 epochs)
#   - val carved out of trainval (80/20), test split untouched until the end
#
# The whole sweep is repeated over N_SEEDS training seeds (seed = 42 + i, the
# repo idiom from gradual_unfreezing.py) so the confusion analysis can report
# mean +/- std and separate stable findings from single-seed noise. Per-run
# test-set predictions are dumped to PREDS_DIR for post-hoc analysis, and
# analysis/confusion.py is invoked at the end.


# -----------------------
# Config
# -----------------------
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll",
    "Russian Blue", "Siamese", "Sphynx",
}
IMBALANCE_RATIO = 0.20                  # keep 20% of each cat breed in train
# Linear probe: freeze the entire backbone and train only the fc head. The
# set_finetune_l_layers() helper counts conv blocks unfrozen *beyond* fc, so
# L = 0 means "fc only" — this is the report's "L = 1" (one trainable layer).
L            = 0
NUM_EPOCHS   = 12
LR           = 1e-3
L1_LAMBDA    = 1e-5                     # L1 penalty on trainable params (report regime)
BATCH_SIZE   = 64
SPLIT_SEED   = 42                       # train/val split — fixed across training seeds
IMBAL_SEED   = 123                      # which cat images survive the 20% cut — fixed
BASE_SEED    = 42                       # training seed for run i is BASE_SEED + i
N_SEEDS      = 5                        # repeat the whole sweep over this many seeds

# Where per-run test predictions and the confusion figures are written. Mirrors
# the repo's dedicated-results-dir convention (cf. ../results_limited_data);
# paths are relative to src/ (the directory this script is run from).
RESULTS_DIR  = "../../data/figures/imbalanced"
PREDS_DIR    = "../../data/figures/imbalanced/preds"


# -----------------------
# Transforms — pulled from limited_data.py (coauthor pipeline).
# Plain pipeline for val/test, aug pipeline for train.
# -----------------------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

data_augmentation = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# -----------------------
# Dataset — two parallel copies of trainval (plain + aug) so the same indices
# can address either transform. Test always uses the plain pipeline.
# -----------------------
data_dir = "../.."

full_train = datasets.OxfordIIITPet(
    root=data_dir, split="trainval",
    transform=data_transform, download=True, target_types="category",
)
full_train_aug = datasets.OxfordIIITPet(
    root=data_dir, split="trainval",
    transform=data_augmentation, download=True, target_types="category",
)
test = datasets.OxfordIIITPet(
    root=data_dir, split="test",
    transform=data_transform, download=True, target_types="category",
)

class_names = full_train.classes
num_classes = len(class_names)

# Cat breed → label index. torchvision sorts class names alphabetically, so a
# fixed range(12) doesn't work — cats and dogs are interleaved.
CAT_CLASS_INDICES = [i for i, n in enumerate(class_names) if n in CAT_BREEDS]
assert len(CAT_CLASS_INDICES) == 12


# -----------------------
# 80/20 trainval → train / val (same idiom as the coauthor scripts).
# trainval is balanced (~99 imgs per class), so random_split is approximately
# stratified; val keeps its balance which is what we want for an honest signal.
# Apply the cat-breed imbalance only to the train half.
# -----------------------
train_size = int(0.8 * len(full_train))
val_size   = len(full_train) - train_size
train_clean, val_set = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(SPLIT_SEED),
)

# Reduce each cat breed to IMBALANCE_RATIO of its samples within train.
_rng = np.random.default_rng(IMBAL_SEED)
_by_class = collections.defaultdict(list)
for idx in train_clean.indices:
    _by_class[full_train._labels[idx]].append(idx)

imbal_idxs = []
for cls, idxs in _by_class.items():
    if cls in CAT_CLASS_INDICES:
        n_keep = max(1, int(len(idxs) * IMBALANCE_RATIO))
        imbal_idxs.extend(_rng.choice(idxs, size=n_keep, replace=False).tolist())
    else:
        imbal_idxs.extend(idxs)

# Training subset uses augmented dataset; val/test use plain.
imbal_train = Subset(full_train_aug, imbal_idxs)
train_labels = [full_train._labels[i] for i in imbal_idxs]

dataset_sizes = {
    "train": len(imbal_train),
    "val":   len(val_set),
    "test":  len(test),
}
print(f"train (imbalanced): {dataset_sizes['train']}  "
      f"val: {dataset_sizes['val']}  test: {dataset_sizes['test']}")


# -----------------------
# Device
# -----------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}")


# -----------------------
# Inverse-frequency weights — used by weighted_ce and the oversampling sampler.
# Normalised so total weight = num_classes (keeps loss scale ~ unweighted CE).
# -----------------------
counts = np.bincount(train_labels, minlength=num_classes).astype(float)
counts_safe = np.where(counts == 0, 1, counts)

ce_weights = 1.0 / counts_safe
ce_weights = ce_weights / ce_weights.sum() * num_classes
ce_weights = torch.tensor(ce_weights, dtype=torch.float).to(device)

sample_weights = torch.tensor(
    [1.0 / counts_safe[l] for l in train_labels], dtype=torch.float
)


# -----------------------
# Model — ResNet-34, fine-tune last l blocks + fc (same helper as
# fine_tune_l_layers.py).
# -----------------------
def set_finetune_l_layers(model, l):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for i in range(l):
        for param in layers[i].parameters():
            param.requires_grad = True


def build_model():
    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    set_finetune_l_layers(model, L)
    return model


def set_seed(seed):
    """Seed every RNG the training run touches (repo idiom, cf.
    gradual_unfreezing.py). Reset before each (seed, strategy) run so the
    strategies share the same head init and augmentation stream at a given
    seed and differ only in loss/sampler."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
# Training loop — coauthor style (limited_data.py): tracks both loss and acc
# histories, steps the scheduler after the train phase, saves best-by-val-acc
# to a TemporaryDirectory so disk doesn't fill with 4 checkpoints.
# -----------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    train_loss_h, val_loss_h = [], []
    train_acc_h,  val_acc_h  = [], []

    with TemporaryDirectory() as tmp:
        best_path = os.path.join(tmp, "best.pt")
        torch.save(model.state_dict(), best_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 20)

            for phase in ["train", "val"]:
                model.train() if phase == "train" else model.eval()
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = outputs.argmax(1)
                        if phase == "train":
                            # L1 penalty on the trainable head (report regime).
                            if L1_LAMBDA > 0:
                                l1 = sum(p.abs().sum()
                                         for p in model.parameters()
                                         if p.requires_grad)
                                loss = loss + L1_LAMBDA * l1
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels).sum().item()

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc  = running_corrects / dataset_sizes[phase]

                if phase == "train":
                    train_loss_h.append(epoch_loss)
                    train_acc_h.append(epoch_acc)
                else:
                    val_loss_h.append(epoch_loss)
                    val_acc_h.append(epoch_acc)

                print(f"  {phase} loss {epoch_loss:.4f}  acc {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_path)

        elapsed = time.time() - since
        print(f"\ndone in {elapsed // 60:.0f}m {elapsed % 60:.0f}s — "
              f"best val acc {best_acc:.4f}")
        model.load_state_dict(torch.load(best_path, weights_only=True))

    return model, train_loss_h, val_loss_h, train_acc_h, val_acc_h


# -----------------------
# Per-class evaluation — task asks for measures beyond overall accuracy.
# -----------------------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            preds = model(inputs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "acc":          (all_preds == all_labels).mean(),
        "f1_per_class": f1_score(all_labels, all_preds, average=None,
                                 labels=list(range(num_classes)),
                                 zero_division=0),
        "f1_macro":     f1_score(all_labels, all_preds, average="macro",
                                 zero_division=0),
        "report":       classification_report(all_labels, all_preds,
                                              target_names=class_names,
                                              zero_division=0),
        # Raw labels/predictions for the post-hoc confusion analysis.
        "y_true":       all_labels,
        "y_pred":       all_preds,
    }


# -----------------------
# Sweep — the whole 4-strategy comparison is repeated over N_SEEDS training
# seeds. For each (seed, strategy) we dump the test-set predictions so the
# post-hoc confusion analysis can run without retraining, and accumulate
# per-strategy curve/metric histories (one entry per seed) for the summary
# figures, which are then averaged across seeds.
# -----------------------
os.makedirs(PREDS_DIR, exist_ok=True)

val_loader  = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test,    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

strategies = ["baseline", "weighted_ce", "oversampling", "weighted_ce+oversampling"]
seeds = [BASE_SEED + i for i in range(N_SEEDS)]

# meta.npz lets analysis/confusion.py map class index -> breed name and know
# which classes are the (reduced) cats — sourced entirely from the dataset
# object, never hardcoded.
np.savez(
    os.path.join(PREDS_DIR, "meta.npz"),
    class_names=np.array(class_names),
    cat_class_indices=np.array(CAT_CLASS_INDICES),
    seeds=np.array(seeds),
    strategies=np.array(strategies),
)

# Per-strategy accumulators — one entry per seed.
runs = {strat: {k: [] for k in
        ("train_acc", "val_acc", "train_loss", "val_loss",
         "test_acc", "test_f1", "val_acc_best", "val_f1", "f1_per_class")}
        for strat in strategies}

for seed in seeds:
    for strat in strategies:
        print(f"\n{'='*45}\nseed {seed} | strategy: {strat}\n{'='*45}")
        # Reset the RNG so every strategy at this seed shares the same head init
        # and augmentation stream, differing only in loss weighting / sampler.
        set_seed(seed)

        use_weights = "weighted_ce"  in strat
        use_sampler = "oversampling" in strat

        if use_sampler:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(imbal_train, batch_size=BATCH_SIZE,
                                      sampler=sampler, num_workers=0)
        else:
            train_loader = DataLoader(imbal_train, batch_size=BATCH_SIZE,
                                      shuffle=True, num_workers=0)

        criterion = (nn.CrossEntropyLoss(weight=ce_weights)
                     if use_weights else nn.CrossEntropyLoss())

        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        model = build_model()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=LR)      # plain Adam + L1 penalty (report regime)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS,
        )

        val_res  = evaluate(model, val_loader)
        test_res = evaluate(model, test_loader)
        print(f"\n[val ] acc {val_res['acc']:.4f}  macro-F1 {val_res['f1_macro']:.4f}")
        print(f"[test] acc {test_res['acc']:.4f}  macro-F1 {test_res['f1_macro']:.4f}")

        # Dump raw test predictions for the confusion analysis (post-hoc).
        np.savez(
            os.path.join(PREDS_DIR, f"{strat}__seed{seed}.npz"),
            y_true=test_res["y_true"],
            y_pred=test_res["y_pred"],
        )

        r = runs[strat]
        r["train_acc"].append(tr_acc);      r["val_acc"].append(va_acc)
        r["train_loss"].append(tr_loss);    r["val_loss"].append(va_loss)
        r["test_acc"].append(test_res["acc"]);      r["test_f1"].append(test_res["f1_macro"])
        r["val_acc_best"].append(val_res["acc"]);   r["val_f1"].append(val_res["f1_macro"])
        r["f1_per_class"].append(test_res["f1_per_class"])


# Seed-averaged views for the summary figures.
def _mean_over_seeds(list_of_arrays):
    return np.mean(np.array(list_of_arrays), axis=0)

avg = {strat: {
    "train_acc":    _mean_over_seeds(runs[strat]["train_acc"]),
    "val_acc":      _mean_over_seeds(runs[strat]["val_acc"]),
    "train_loss":   _mean_over_seeds(runs[strat]["train_loss"]),
    "val_loss":     _mean_over_seeds(runs[strat]["val_loss"]),
    "f1_per_class": _mean_over_seeds(runs[strat]["f1_per_class"]),
} for strat in strategies}


# -----------------------
# Plot 1: train/val accuracy curves (mean over seeds)
# -----------------------
plt.figure(figsize=(10, 6))
for strat in strategies:
    r = avg[strat]
    ep = range(1, len(r["val_acc"]) + 1)
    plt.plot(ep, r["train_acc"], "--", alpha=0.7, label=f"train {strat}")
    plt.plot(ep, r["val_acc"], label=f"val {strat}")
plt.title(f"Imbalance mitigation — train/val accuracy (mean over {N_SEEDS} seeds)")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "compare_imbalance_strategies.png"))
plt.close()


# -----------------------
# Plot 2: train/val loss curves (mean over seeds)
# -----------------------
plt.figure(figsize=(10, 6))
for strat in strategies:
    r = avg[strat]
    ep = range(1, len(r["val_loss"]) + 1)
    plt.plot(ep, r["train_loss"], "--", alpha=0.7, label=f"train {strat}")
    plt.plot(ep, r["val_loss"], label=f"val {strat}")
plt.title(f"Imbalance mitigation — train/val loss (mean over {N_SEEDS} seeds)")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "compare_imbalance_loss.png"))
plt.close()


# -----------------------
# Plot 3: per-class F1 on test, mean over seeds (cat columns marked)
# -----------------------
x = np.arange(num_classes)
width = 0.20
fig, ax = plt.subplots(figsize=(18, 5))
for i, strat in enumerate(strategies):
    ax.bar(x + i * width, avg[strat]["f1_per_class"], width, label=strat)
for c in CAT_CLASS_INDICES:
    ax.axvline(c + 1.5 * width, color="navy", linewidth=0.6, alpha=0.4)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.set_title(f"Per-class F1 by strategy, mean over {N_SEEDS} seeds  "
             "(navy lines = cat breeds, reduced to 20%)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "per_class_f1.png"))
plt.close()


# -----------------------
# Summary — mean +/- std across seeds (overall gains are tiny, so the spread
# matters as much as the mean).
# -----------------------
print("\n" + "=" * 74)
print(f"SUMMARY — mean +/- std over {N_SEEDS} seeds")
print("=" * 74)
print(f"{'strategy':<28}  {'test_acc':>15}  {'test_F1':>15}  {'val_acc':>15}")
for strat in strategies:
    r = runs[strat]
    ta, tf = np.array(r["test_acc"]), np.array(r["test_f1"])
    va = np.array(r["val_acc_best"])
    print(f"{strat:<28}  {ta.mean():>7.4f}+/-{ta.std():<6.4f}  "
          f"{tf.mean():>7.4f}+/-{tf.std():<6.4f}  "
          f"{va.mean():>7.4f}+/-{va.std():<6.4f}")


# -----------------------
# Post-hoc confusion analysis (deliverables in analysis/confusion.py). Runs on
# the dumped predictions, so it can equally be re-run standalone:
#     python -m analysis.confusion ../../data/figures/imbalanced/preds ../../data/figures/imbalanced
# -----------------------
try:
    from analysis.confusion import run_confusion_analysis
    run_confusion_analysis(preds_dir=PREDS_DIR, out_dir=RESULTS_DIR)
except Exception as exc:  # analysis must never sink a completed training run
    print(f"\n[confusion analysis skipped: {exc}]")
    print(f"Predictions are saved in {PREDS_DIR}; re-run the analysis with:")
    print(f"    python -m analysis.confusion {PREDS_DIR} {RESULTS_DIR}")