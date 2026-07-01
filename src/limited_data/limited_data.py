# --- run from anywhere (e.g. VS Code Code Runner): resolve data paths and
# sibling imports relative to THIS file, not the caller's working dir ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_HERE)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
# --- end launch guard ---

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.model_selection import train_test_split



# In[2]:


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_augmentation = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[3]:


data_dir = "../.."
full_train = datasets.OxfordIIITPet(root=data_dir,split="trainval", transform=data_transform, download=False, target_types="category")
test = datasets.OxfordIIITPet(root=data_dir,split="test", transform=data_transform, download=False, target_types="category")


# In[4]:


class_names = full_train.classes

def get_limited_dataloaders(train_dataset, fraction, batch_size=4):
    if fraction == 1.0:
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    labels = [train_dataset.dataset._labels[i] for i in train_dataset.indices]
    selected, _ = train_test_split(train_dataset.indices, train_size=fraction, stratify=labels)
    limited_data = Subset(train_dataset.dataset, selected)
    return DataLoader(limited_data, batch_size=batch_size, shuffle=True, num_workers=0)


train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train, val = random_split(full_train, [train_size, val_size])

full_train_aug = datasets.OxfordIIITPet(root=data_dir,split="trainval", transform=data_augmentation, download=False, target_types="category")
train_aug = Subset(full_train_aug, train.indices)

val_loader = DataLoader(val, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test, batch_size=4, shuffle=False, num_workers=0)


dataloaders = []
fractions = [0.02, 0.1, 0.5, 1.0]
for frac in fractions:
    train_loader = get_limited_dataloaders(train_aug, frac)
    dataloaders.append({"train": train_loader, "val": val_loader, "test": test_loader})


# In[5]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")


# In[6]:


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()


                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


# In[7]:


def get_model():
    model = models.resnet34(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(in_features=512, out_features=37)
    return model.to(device)

model = get_model()

criterion = torch.nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model.fc.parameters(), lr=0.001)


# In[ ]:


results = []
for i, fraction in enumerate(fractions):
    print(f"Training with {int(fraction*100)}% of the training data")
    model = get_model()
    optimizer_ft = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    dataloader = dataloaders[i]
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(model, dataloader, criterion, optimizer_ft, scheduler, num_epochs=10)

    results.append((fraction, train_loss_history, val_loss_history, train_acc_history, val_acc_history))


# In[ ]:


fracs = [r[0] for r in results]
test_accs = [r[4] for r in results]

plt.figure(figsize=(10, 5))
for frac, _, _, train_acc, val_acc in results:
    plt.plot(train_acc, linestyle='--', label=f'Train {int(frac*100)}%')
    plt.plot(val_acc, linestyle='-', label=f'Val {int(frac*100)}%')
plt.title('Train and Val Accuracy vs Data fractions')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(title='Fraction', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('../../data/figures/limited_data/aug_accuracy_limited_data.png')  # Save the plot as a PNG file
plt.show()

# Val Loss
plt.figure(figsize=(10, 5))
for frac, train_loss, val_loss, _, _ in results:
    plt.plot(train_loss, linestyle='--', label=f'Train {int(frac*100)}%')
    plt.plot(val_loss, linestyle='-', label=f'Val {int(frac*100)}%')
plt.title('Train and Val Loss vs Data fractions')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(title='Fraction', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('../../data/figures/limited_data/aug_loss_limited_data.png')  # Save the plot as a PNG file
plt.show()

