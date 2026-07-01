# --- run from anywhere (e.g. VS Code Code Runner): resolve data paths and
# sibling imports relative to THIS file, not the caller's working dir ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_HERE)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
# --- end launch guard ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch.utils.data import random_split

# -----------------------
# Transform
# -----------------------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------
# Dataset
# -----------------------
data_dir = "../.."

full_train = datasets.OxfordIIITPet(
    root=data_dir,
    split="trainval",
    transform=data_transform,
    download=True,
    target_types="binary-category"
)

test = datasets.OxfordIIITPet(
    root=data_dir,
    split="test",
    transform=data_transform,
    download=True,
    target_types="binary-category"
)

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train, val = random_split(full_train, [train_size, val_size])

dataloaders = {
    "train": DataLoader(train, batch_size=32, shuffle=True,  num_workers=0),
    "val":   DataLoader(val,   batch_size=32, shuffle=False, num_workers=0),
    "test":  DataLoader(test,  batch_size=32, shuffle=False, num_workers=0),
}

dataset_sizes = {
    "train": len(train),
    "val":   len(val),
    "test":  len(test)
}
class_names = ["Cat", "Dog"]

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
# Visualize batch
# -----------------------
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

inputs, classes = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

# -----------------------
# Training loop
# -----------------------
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=5):
    since = time.time()

    best_model_path = "best_model.pt"
    best_acc = 0.0

    torch.save(model.state_dict(), best_model_path)

    train_acc_list = []
    val_acc_list = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 20)

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

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == "train":
                train_acc_list.append(epoch_acc)
            else:
                val_acc_list.append(epoch_acc)

            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)

    time_elapsed = time.time() - since
    print(f"\nTraining done in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    return model, train_acc_list, val_acc_list

# -----------------------
# Visualization
# -----------------------
def visualize_model(model, num_images=6):
    model.eval()
    shown = 0

    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(1)

            for i in range(inputs.size(0)):
                shown += 1

                plt.subplot(num_images // 2, 2, shown)
                plt.axis("off")
                plt.title(f"pred: {class_names[preds[i]]}")

                imshow(inputs.cpu()[i])

                if shown == num_images:
                    return

def evaluate_on_test(model):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            running_corrects += (preds == labels).sum().item()

    acc = running_corrects / dataset_sizes["test"]
    print(f"Test accuracy: {acc:.4f}")
    return acc

# -----------------------
# Model (freeze all layers except final, to fine-tune last layer)
# -----------------------
num_runs = 5
all_train_acc = []
all_val_acc = []
all_test_acc = []
for run in range(num_runs):
  print(f"\n======== RUN {run+1}/{num_runs} ========")

  model = models.resnet34(weights="IMAGENET1K_V1")
  model.fc = nn.Linear(model.fc.in_features, 2)

  #Freeze all pretrained layers
  for param in model.parameters():
      param.requires_grad = False
  #Unfreeze final classification layer
  for param in model.fc.parameters():
      param.requires_grad = True

  model = model.to(device)

  # -----------------------
  # Loss + Optimizer
  # -----------------------
  # model.fc = final fully connected classification layer (replaces original ResNet classifier)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

  model, train_acc, val_acc = train_model(model, criterion, optimizer, scheduler=None, num_epochs=5)

  test_acc = evaluate_on_test(model)
  all_train_acc.append(train_acc)
  all_val_acc.append(val_acc)
  all_test_acc.append(test_acc)

#average
avg_train_acc = np.mean(all_train_acc, axis=0)
avg_val_acc = np.mean(all_val_acc, axis=0)
avg_test_acc = np.mean(all_test_acc)

epochs = range(1, len(avg_train_acc) + 1)

plt.figure()
plt.plot(epochs, avg_train_acc, label="train", linestyle="--")
plt.plot(epochs, avg_val_acc, label="val")
plt.axhline(
    y=avg_test_acc,
    color="red",
    linestyle=":",
    label=f"test ({avg_test_acc:.4f})"
)

plt.title("Average Accuracy over 5 Runs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("../../data/figures/binary_classification/accuracy.png")
plt.show()
plt.close()