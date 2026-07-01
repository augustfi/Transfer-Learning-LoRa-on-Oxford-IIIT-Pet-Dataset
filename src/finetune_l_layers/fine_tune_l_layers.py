import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, models, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time


# Strategy 1; Fine-tune l layers simultaneously Fine-tune the last l layers of the
# network (+ the classification layer) from the start of training. For the first experiment
# set l = 1, the next one re-fine-tune the pre-trained network but with l = 2, then l = 3
# until l = L where L is defined by your available compute and also seeing when adding
# more layers results in only minimal or no changes.



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
# Dataset (MULTI-CLASS)
# -----------------------
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

# Split trainval → 80% train, 20% val
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train, val = random_split(full_train, [train_size, val_size])


#num_workers=0 for macOS/Python 3.14, otherwise issue for some reason
dataloaders = {
    "train": DataLoader(train, batch_size=64, shuffle=True,  num_workers=0),
    "val":   DataLoader(val,   batch_size=64, shuffle=False, num_workers=0),
    "test":  DataLoader(test,  batch_size=64, shuffle=False, num_workers=0),
}

dataset_sizes = {
    "train": len(train),
    "val": len(val),
    "test": len(test)
}

# 37 breed classes, instead of cat/dog
class_names = full_train.classes

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

inputs, labels = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in labels])

# -----------------------
# Training loop
# -----------------------
def train_model(model, criterion, optimizer, num_epochs=5, l=1):
    since = time.time()

    best_model_path = f"best_multiclass_model_l{l}.pt"
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

    time_elapsed = time.time() - since
    print(f"\nTraining done in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    return model, train_acc_list, val_acc_list

# -----------------------
# Model (37 classes) (freeze all layers except final, to fine-tune last layer)
# -----------------------
def set_finetune_l_layers(model, l):
    #freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    #always unfreeze classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    #ResNet blocks in order from deep layers to shallow
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]

    #unfreeze last l blocks
    for i in range(l):
        for param in layers[i].parameters():
            param.requires_grad = True

def describe_layers(l):
    names = ["layer4", "layer3", "layer2", "layer1"]
    return ["fc"] + names[:l]

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
    print(f"  Test accuracy: {acc:.4f}")
    return acc

all_results = {} # for plotting accuracy graph
max_l = 4
num_runs = 5
for l in range(1, max_l + 1):
    print(f"\nTraining with l = {l}")
    all_train_runs = []
    all_val_runs = []
    test_accs = []

    for run in range(num_runs):
      print(f"Run {run+1}/{num_runs}")

      model = models.resnet34(weights="IMAGENET1K_V1")
      model.fc = nn.Linear(model.fc.in_features, 37)
      model = model.to(device)

      #unfreeze l layers
      set_finetune_l_layers(model, l)

      print("Trainable parts:", describe_layers(l))
      print("Trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

      #Loss + optimizer
      criterion = nn.CrossEntropyLoss()

      #optimizer uses all trainable parameters (those not frozen)
      trainable_params = [p for p in model.parameters() if p.requires_grad]
      optimizer = optim.Adam(trainable_params, lr=0.001)

      #Train
      model, train_acc, val_acc = train_model(
          model,
          criterion,
          optimizer,
          num_epochs=5,
          l=l
      )
      print(f"Evaluating best model for l={l} on test set...")
      test_acc = evaluate_on_test(model)
      #all_results[l] = (train_acc, val_acc)
      all_train_runs.append(train_acc)
      all_val_runs.append(val_acc)
      test_accs.append(test_acc)

    # average over runs
    avg_train = np.mean(np.stack(all_train_runs), axis=0)
    avg_val = np.mean(np.stack(all_val_runs), axis=0)
    avg_test = np.mean(test_accs)
    all_results[l] = (avg_train, avg_val, avg_test)
    print(f"Avg test accuracy over {num_runs} runs: {avg_test:.4f}")

plt.figure()
for l, (train_acc, val_acc, test_acc) in all_results.items():
    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, linestyle="--", label=f"train l={l}")
    plt.plot(epochs, val_acc, label=f"val l={l}")

    plt.axhline(
        y=test_acc,
        linestyle=":",
        alpha=0.5,
        label=f"test l={l}"
    )

plt.title("Fine-tuning depth comparison (avg over 5 runs)")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("../../data/figures/finetune_l_layers/compare_all_l_avg.png")
plt.close()