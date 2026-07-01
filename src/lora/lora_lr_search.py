# --- run from anywhere (e.g. VS Code Code Runner): resolve data paths and
# sibling imports relative to THIS file, not the caller's working dir ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_HERE)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
# --- end launch guard ---

# Learning-rate sweep with cosine annealing.
# Self-contained: sets up data, model builder, and a local training loop with scheduler.step().

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import time

from lora_finetune import build_lora_resnet50

# ── data setup ───────────────────────────────────────────────────────────────
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = "../.."

full_train = datasets.OxfordIIITPet(
    root=data_dir, split="trainval", transform=data_transform,
    download=True, target_types="category",
)
test = datasets.OxfordIIITPet(
    root=data_dir, split="test", transform=data_transform,
    download=True, target_types="category",
)

train_size = int(0.85 * len(full_train))
val_size = len(full_train) - train_size
train_set, val_set = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

dataloaders = {
    "train": DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0),
    "val":   DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0),
    "test":  DataLoader(test,      batch_size=32, shuffle=False, num_workers=0),
}
dataset_sizes = {"train": len(train_set), "val": len(val_set)}

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}")


def eval_test(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


R = 8
ALPHA = 16  # α/r = 2 (matches rank-sweep convention; adjust once α sweep is settled)
EPOCHS = 15


def train_with_scheduler(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_acc = 0.0
    best_path = "best_lora_model.pt"
    torch.save(model.state_dict(), best_path)

    train_acc_list, val_acc_list, lr_list = [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}  lr={optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 20)
        lr_list.append(optimizer.param_groups[0]["lr"])

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            (train_acc_list if phase == "train" else val_acc_list).append(epoch_acc)
            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_path)

        scheduler.step()  # cosine decay step, once per epoch

    elapsed = time.time() - since
    print(f"\nTraining done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s | best val acc: {best_acc:.4f}")
    model.load_state_dict(torch.load(best_path))
    return model, train_acc_list, val_acc_list, lr_list


configs = [
    {"lr": 1e-4, "label": "peak lr = 1e-4"},
    {"lr": 3e-4, "label": "peak lr = 3e-4"},
    {"lr": 1e-3, "label": "peak lr = 1e-3"},
    {"lr": 3e-3, "label": "peak lr = 3e-3"},
]

epochs_range = range(1, EPOCHS + 1)
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (cfg, color) in enumerate(zip(configs, colors)):
    # ── reproducibility ──────────────────────────────────────────────────────
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── build model ──────────────────────────────────────────────────────────
    model = build_lora_resnet50(
        num_classes=37,
        r=R,
        alpha=ALPHA,
        dropout=0.0,
        target_names=["conv3"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

    print(f"\n{'='*50}")
    print(f"Running: {cfg['label']}")
    print(f"{'='*50}")

    # ── train ────────────────────────────────────────────────────────────────
    model, train_acc, val_acc, lr_curve = train_with_scheduler(
        model, criterion, optimizer, scheduler, num_epochs=EPOCHS
    )

    test_acc = eval_test(model, dataloaders["test"], device)
    print(f"Test accuracy: {test_acc:.4f}")

    # ── plot: train + val in same subplot ────────────────────────────────────
    ax = axes[i]
    ax.plot(epochs_range, train_acc, color=color, linestyle="--", label="train")
    ax.plot(epochs_range, val_acc,   color=color, linestyle="-",  label="val")
    ax.set_title(f"{cfg['label']}  |  test={test_acc:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(1, EPOCHS)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(f"LoRA LR sweep + cosine annealing  (r={R}, α={ALPHA}, conv3 only)\n-- train   — val", fontsize=13)
plt.tight_layout()
plt.savefig("../../data/figures/lora/lora_lr_sweep.png", dpi=150)
plt.show()
print("Saved lora_lr_sweep.png")
