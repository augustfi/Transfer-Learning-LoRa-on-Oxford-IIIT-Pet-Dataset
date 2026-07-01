import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
from lora_finetune import build_lora_resnet50, build_full_finetune, build_linear_probe
import os
results_dir = "lora_rank_sweep"
os.makedirs(results_dir, exist_ok=True)

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


data_dir = "../.."

full_train = datasets.OxfordIIITPet(
    root=data_dir,
    split="trainval",
    transform=data_transform,
    download=True,
    target_types="category"
)

test = datasets.OxfordIIITPet(
    root=data_dir,
    split="test",
    transform=data_transform,
    download=True,
    target_types="category"
)

train_size = int(0.85 * len(full_train))
val_size = len(full_train) - train_size

train_set, val_set = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

dataloaders = {
    "train": DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0),
    "val":   DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0),
    "test":  DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
}

dataset_sizes = {
    "train": len(train_set),
    "val": len(val_set),
}

class_names = full_train.classes


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device}")

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.figure()
    plt.imshow(inp)
    if title is not None:
        plt.title(str(title))
    plt.show()

inputs, labels = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in labels])


def train_model(model, criterion, optimizer, num_epochs):
    since = time.time()

    best_acc = 0.0
    best_model_path = f"best_lora_model.pt"
    torch.save(model.state_dict(), best_model_path)

    train_acc_list = []
    val_acc_list = []
    epoch_times = []
    epoch_peak_memory = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 20)

        print("Trainable parameters:",
            sum(p.numel() for p in model.parameters() if p.requires_grad))

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
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            #plotting accuracy
            if phase == "train":
                train_acc_list.append(epoch_acc)
            else:
                val_acc_list.append(epoch_acc)

            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
            
            epoch_times.append(time.time() - epoch_start)
            if device == "cuda":
                epoch_peak_memory.append(torch.cuda.max_memory_allocated() / 1e9) 

    time_elapsed = time.time() - since
    print(f"\nTraining done in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    return model, train_acc_list, val_acc_list, epoch_times, epoch_peak_memory


def eval_test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main(mode="lora", rank=8, seed=42, run_id=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if mode == "lora":
        model = build_lora_resnet50(
            num_classes=37,
            r=rank,
            alpha=2 * rank,
            dropout=0.0,
            target_names=("conv1", "conv2", "conv3"),
        )
    elif mode == "linear_probe":
        model = build_linear_probe(num_classes=37)
    elif mode == "full_finetune":
        model = build_full_finetune(num_classes=37)
    model = model.to(device)

    epochs = 15

    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    model, train_acc, val_acc, epoch_times, epoch_peak_memory = train_model(
        model,
        criterion,
        optimizer,
        epochs,
    )

    test_acc = eval_test(model, dataloaders["test"], device)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    plt.figure()
    epochs_range = range(1, len(val_acc) + 1)
    plt.plot(epochs_range, train_acc, label="train", linestyle="--")
    plt.plot(epochs_range, val_acc, label="val")
    plt.title(f"{mode} rank={rank} run={run_id}: training and validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{mode}_rank{rank}_run{run_id}.png"))
    plt.close()

    return test_acc, train_acc, val_acc

if __name__ == "__main__":
    ranks = [8, 16, 32]
    seeds = [42, 123, 2024]  # 3 runs per rank

    # store all curves for the combined plot
    all_results = {r: {"train": [], "val": [], "test_accs": []} for r in ranks}

    for rank in ranks:
        for run_id, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"LoRA rank={rank}, run={run_id}, seed={seed}")
            print(f"{'='*60}")
            test_acc, train_acc, val_acc = main(
                mode="lora", rank=rank, seed=seed, run_id=run_id
            )
            all_results[rank]["train"].append(train_acc)
            all_results[rank]["val"].append(val_acc)
            all_results[rank]["test_accs"].append(test_acc)

    # report average test accuracy per rank
    print("\n" + "=" * 60)
    print("SUMMARY: average test accuracy per rank")
    print("=" * 60)
    for rank in ranks:
        accs = all_results[rank]["test_accs"]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"rank={rank}: test_accs={[f'{a:.4f}' for a in accs]} "
              f"| mean={mean_acc:.4f} std={std_acc:.4f}")

    # combined plot: all train and val curves across all ranks and runs
    plt.figure(figsize=(10, 6))
    colors = {8: "tab:blue", 16: "tab:orange", 32: "tab:green"}

    for rank in ranks:
        for run_id in range(len(seeds)):
            train_curve = all_results[rank]["train"][run_id]
            val_curve = all_results[rank]["val"][run_id]
            epochs_range = range(1, len(val_curve) + 1)

            # only label the first run of each rank so the legend isn't cluttered
            train_label = f"rank={rank} train" if run_id == 0 else None
            val_label = f"rank={rank} val" if run_id == 0 else None

            plt.plot(epochs_range, train_curve, color=colors[rank],
                     linestyle="--", alpha=0.7, label=train_label)
            plt.plot(epochs_range, val_curve, color=colors[rank],
                     linestyle="-", alpha=0.9, label=val_label)

    plt.title("LoRA rank sweep: train (dashed) and val (solid) accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "rank_sweep_combined.png"), dpi=150)
    plt.close()

    print(f"\nAll figures saved to '{results_dir}/'")