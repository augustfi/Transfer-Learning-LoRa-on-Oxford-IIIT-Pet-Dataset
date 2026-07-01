import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from lora_finetune import build_lora_resnet50

results_dir = "../../data/figures/lora/full_vs_lora"
os.makedirs(results_dir, exist_ok=True)


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data_dir = "../.."

full_train = datasets.OxfordIIITPet(
    root=data_dir, split="trainval",
    transform=data_transform, download=True, target_types="category"
)
test = datasets.OxfordIIITPet(
    root=data_dir, split="test",
    transform=data_transform, download=True, target_types="category"
)

class_names = full_train.classes

def get_limited_dataloader(train_subset, fraction, batch_size=4, seed=42):
    if fraction == 1.0:
        return DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    labels = [train_subset.dataset._labels[i] for i in train_subset.indices]
    selected, _ = train_test_split(
        train_subset.indices, train_size=fraction,
        stratify=labels, random_state=seed
    )
    limited = Subset(train_subset.dataset, selected)
    return DataLoader(limited, batch_size=batch_size, shuffle=True, num_workers=0)



train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_split, val_split = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_split, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test, batch_size=4, shuffle=False, num_workers=0)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_acc = 0.0
    best_model_path = "best_model_limited.pt"
    torch.save(model.state_dict(), best_model_path)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history



def eval_test(model, dataloader, device):
    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def get_full_finetune_resnet50(num_classes=37):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def get_lora_resnet50(num_classes=37, r=8, alpha=16):
    model = build_lora_resnet50(
        num_classes=num_classes,
        r=r,
        alpha=alpha,
        dropout=0.0,
        target_names=("conv3",),
    )
    return model.to(device)



def run_one(method, fraction, seed=42, num_epochs=10, lr=1e-3, weight_decay=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = get_limited_dataloader(train_split, fraction, batch_size=4, seed=seed)
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    print(f"\n[{method}] fraction={fraction}: {len(train_loader.dataset)} train images")

    if method == "lora":
        model = get_lora_resnet50(num_classes=37, r=8, alpha=16)
    elif method == "full_ft":
        model = get_full_finetune_resnet50(num_classes=37)
    else:
        raise ValueError(f"Unknown method: {method}")

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs
    )

    test_acc = eval_test(model, test_loader, device)
    print(f"[{method}] fraction={fraction}: test_acc={test_acc:.4f}")

    plt.figure(figsize=(8, 4))
    epochs_range = range(1, len(val_acc) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label="train", linestyle="--")
    plt.plot(epochs_range, val_acc, label="val")
    plt.title(f"{method} frac={fraction} — accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label="train", linestyle="--")
    plt.plot(epochs_range, val_loss, label="val")
    plt.title(f"{method} frac={fraction} — loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{method}_frac{fraction}_curves.png"))
    plt.close()

    return {
        "test_acc": test_acc,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


if __name__ == "__main__":
    fractions = [0.02, 0.1, 0.5, 1.0]
    methods = ["full_ft", "lora"]
    num_epochs = 15
    lr = 1e-3
    weight_decay = 1e-4

    all_results = {m: {} for m in methods}

    for method in methods:
        for fraction in fractions:
            print(f"\n{'=' * 60}")
            print(f"METHOD={method} | FRACTION={fraction}")
            print(f"{'=' * 60}")
            result = run_one(method, fraction, seed=42,
                             num_epochs=num_epochs, lr=lr,
                             weight_decay=weight_decay)
            all_results[method][fraction] = result

    # Summary printout
    print("\n" + "=" * 60)
    print("SUMMARY: test accuracy by method and data fraction")
    print("=" * 60)
    print(f"{'fraction':<12} {'LoRA test':<15} {'Full-FT test':<15} {'gap (LoRA - FT)':<18}")
    for fraction in fractions:
        lora_acc = all_results["lora"][fraction]["test_acc"]
        ft_acc = all_results["full_ft"][fraction]["test_acc"]
        gap = lora_acc - ft_acc
        print(f"{fraction:<12} {lora_acc:<15.4f} {ft_acc:<15.4f} {gap:+.4f}")

    # Headline plot
    lora_test_accs = [all_results["lora"][f]["test_acc"] for f in fractions]
    ft_test_accs = [all_results["full_ft"][f]["test_acc"] for f in fractions]

    plt.figure(figsize=(8, 5))
    plt.plot([f * 100 for f in fractions], ft_test_accs,
             marker="o", linestyle="-", color="tab:blue", label="Full fine-tuning")
    plt.plot([f * 100 for f in fractions], lora_test_accs,
             marker="s", linestyle="-", color="tab:orange", label="LoRA (conv3, r=8)")
    plt.title("Test accuracy vs training-data fraction (ResNet50, no aug)")
    plt.xlabel("Training data used (%)")
    plt.ylabel("Test accuracy")
    plt.xscale("log")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "headline_test_acc_vs_fraction.png"), dpi=150)
    plt.close()

    # Combined train+val curves across all fractions and methods
    plt.figure(figsize=(12, 6))
    frac_colors = {0.02: "tab:red", 0.1: "tab:orange", 0.5: "tab:green", 1.0: "tab:blue"}

    plt.subplot(1, 2, 1)
    for fraction in fractions:
        train = all_results["lora"][fraction]["train_acc"]
        val = all_results["lora"][fraction]["val_acc"]
        epochs_range = range(1, len(val) + 1)
        plt.plot(epochs_range, train, color=frac_colors[fraction],
                 linestyle="--", alpha=0.6,
                 label=f"{int(fraction * 100)}% train")
        plt.plot(epochs_range, val, color=frac_colors[fraction],
                 linestyle="-",
                 label=f"{int(fraction * 100)}% val")
    plt.title("LoRA (conv3, r=8) — train/val by fraction")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(fontsize=8, ncol=2, loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for fraction in fractions:
        train = all_results["full_ft"][fraction]["train_acc"]
        val = all_results["full_ft"][fraction]["val_acc"]
        epochs_range = range(1, len(val) + 1)
        plt.plot(epochs_range, train, color=frac_colors[fraction],
                 linestyle="--", alpha=0.6,
                 label=f"{int(fraction * 100)}% train")
        plt.plot(epochs_range, val, color=frac_colors[fraction],
                 linestyle="-",
                 label=f"{int(fraction * 100)}% val")
    plt.title("Full FT (ResNet50) — train/val by fraction")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(fontsize=8, ncol=2, loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_train_val_curves.png"), dpi=150)
    plt.close()

    print(f"\nAll figures saved to '{results_dir}/'")
