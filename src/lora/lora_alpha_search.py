# Alpha/r ratio sweep — reuses build_lora_resnet50, train_model, eval_test,
# dataloaders, device already defined in earlier cells.

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

R = 8
EPOCHS = 15

configs = [
    {"alpha":  8, "label": "α/r = 1  (α=8)"},
    {"alpha": 16, "label": "α/r = 2  (α=16)"},
    {"alpha": 32, "label": "α/r = 4  (α=32)"},
    {"alpha": 64, "label": "α/r = 8  (α=64)"},
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
        alpha=cfg["alpha"],
        dropout=0.0,
        target_names=["conv3"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    print(f"\n{'='*50}")
    print(f"Running: {cfg['label']}")
    print(f"{'='*50}")

    # ── train ────────────────────────────────────────────────────────────────
    model, train_acc, val_acc, _, _ = train_model(
        model, criterion, optimizer, num_epochs=EPOCHS
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

fig.suptitle(f"LoRA α/r ratio sweep  (r={R}, conv3 only)\n-- train   — val", fontsize=13)
plt.tight_layout()
plt.savefig("../../data/figures/lora/lora_ratio_sweep.png", dpi=150)
plt.show()
print("Saved lora_ratio_sweep.png")
