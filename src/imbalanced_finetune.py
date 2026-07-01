
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
# AdamW + StepLR) from limited_data.py / gradual_unfreezing.py, and layers
# the imbalance experiment on top so the training regime is identical
# across strategies. The four strategies compared:
#   - baseline                    : plain CE on the imbalanced train set
#   - weighted_ce                 : nn.CrossEntropyLoss(weight = inv-freq)
#   - oversampling                : WeightedRandomSampler with inv-freq
#   - weighted_ce + oversampling  : both at once
#
# Generalization stack (shared by all four strategies, so the comparison is
# apples-to-apples instead of three flavors of overfit):
#   - data augmentation on train (RandomResizedCrop + flip + rotation)
#   - AdamW with weight_decay = 1e-4
#   - StepLR scheduler  (drop LR by 10x every 7 epochs)
#   - val carved out of trainval (80/20), test split untouched until the end


# -----------------------
# Config
# -----------------------
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll",
    "Russian Blue", "Siamese", "Sphynx",
}
IMBALANCE_RATIO = 0.20                  # keep 20% of each cat breed in train
L            = 2                        # fine-tune last l ResNet blocks + fc
NUM_EPOCHS   = 12
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
SPLIT_SEED   = 42
IMBAL_SEED   = 123


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
data_dir = ".."

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
    }


# -----------------------
# Sweep
# -----------------------
val_loader  = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test,    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

strategies = ["baseline", "weighted_ce", "oversampling", "weighted_ce+oversampling"]
all_results = {}

for strat in strategies:
    print(f"\n{'='*45}\nStrategy: {strat}\n{'='*45}")

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
    optimizer = optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, tr_loss, va_loss, tr_acc, va_acc = train_model(
        model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS,
    )

    val_res  = evaluate(model, val_loader)
    test_res = evaluate(model, test_loader)
    print(f"\n[val ] acc {val_res['acc']:.4f}  macro-F1 {val_res['f1_macro']:.4f}")
    print(f"[test] acc {test_res['acc']:.4f}  macro-F1 {test_res['f1_macro']:.4f}")
    print(f"\nClassification report (TEST) — {strat}")
    print(test_res["report"])

    all_results[strat] = {
        "train_acc":    tr_acc,
        "val_acc":      va_acc,
        "train_loss":   tr_loss,
        "val_loss":     va_loss,
        "test_acc":     test_res["acc"],
        "test_f1":      test_res["f1_macro"],
        "val_acc_best": val_res["acc"],
        "val_f1":       val_res["f1_macro"],
        "f1_per_class": test_res["f1_per_class"],
    }


# -----------------------
# Plot 1: train/val accuracy curves
# -----------------------
plt.figure(figsize=(10, 6))
for strat, r in all_results.items():
    ep = range(1, len(r["val_acc"]) + 1)
    plt.plot(ep, r["train_acc"], "--", alpha=0.7, label=f"train {strat}")
    plt.plot(ep, r["val_acc"], label=f"val {strat}")
plt.title("Imbalance mitigation — train/val accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("compare_imbalance_strategies.png")
plt.close()


# -----------------------
# Plot 2: train/val loss curves
# -----------------------
plt.figure(figsize=(10, 6))
for strat, r in all_results.items():
    ep = range(1, len(r["val_loss"]) + 1)
    plt.plot(ep, r["train_loss"], "--", alpha=0.7, label=f"train {strat}")
    plt.plot(ep, r["val_loss"], label=f"val {strat}")
plt.title("Imbalance mitigation — train/val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("compare_imbalance_loss.png")
plt.close()


# -----------------------
# Plot 3: per-class F1 on test (cat columns marked)
# -----------------------
x = np.arange(num_classes)
width = 0.20
fig, ax = plt.subplots(figsize=(18, 5))
for i, strat in enumerate(strategies):
    ax.bar(x + i * width, all_results[strat]["f1_per_class"], width, label=strat)
for c in CAT_CLASS_INDICES:
    ax.axvline(c + 1.5 * width, color="navy", linewidth=0.6, alpha=0.4)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.set_title("Per-class F1 by strategy  (navy lines = cat breeds, reduced to 20%)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("per_class_f1.png")
plt.close()


# -----------------------
# Summary
# -----------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'strategy':<28}  {'test_acc':>9}  {'test_F1':>9}  {'val_acc':>9}  {'val_F1':>9}")
for strat in strategies:
    r = all_results[strat]
    print(f"{strat:<28}  {r['test_acc']:>9.4f}  {r['test_f1']:>9.4f}  "
          f"{r['val_acc_best']:>9.4f}  {r['val_f1']:>9.4f}")