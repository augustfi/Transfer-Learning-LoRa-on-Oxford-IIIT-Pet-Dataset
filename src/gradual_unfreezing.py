import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
data_dir = ".."

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

train_size = int(0.85*len(full_train))
val_size = len(full_train) - train_size

train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

#num_workers=0 for macOS/Python 3.14, otherwise issue for some reason
dataloaders = {
    "train": DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0),
    "val":   DataLoader(val_set, batch_size=64, shuffle=False),
    "test": DataLoader(test, batch_size=64, shuffle=False, num_workers=0)
}

dataset_sizes = {
    "train": len(train_set),
    "val": len(val_set),
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


#schedule for gradual unfreezing
def schedule(num_epochs, epoch, epochs_per_stage, l_max):
    layers = min(l_max, epoch// epochs_per_stage + 1)
    update_step = epoch// epochs_per_stage + 1
    
    return layers, update_step
# -----------------------
# Training loop
# -----------------------



def set_finetune_l_layers(model, l):
    #freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    #always unfreeze classifier
    ##for param in model.fc.parameters():
        #param.requires_grad = True

    #ResNet blocks in order from deep layers to shallow
    layers = [model.fc, model.layer4, model.layer3, model.layer2, model.layer1]

    #unfreeze last l blocks
    for i in range(l):
        for param in layers[i].parameters():
            param.requires_grad = True
            

def describe_layers(l):
    names = ["model.fc", "layer4", "layer3", "layer2", "layer1"]
    return names[:l]


def build_optimizer(model, num_layers, weight_decay=1e-4):
    param_groups = []
    param_groups.append({'params': model.fc.parameters(), 'lr': 1e-3})
    if num_layers >= 2:
        param_groups.append({'params': model.layer4.parameters(), 'lr': 1e-4})
    if num_layers >= 3:
        param_groups.append({'params': model.layer3.parameters(), 'lr': 1e-4})
    if num_layers >= 4:
        param_groups.append({'params': model.layer2.parameters(), 'lr': 1e-5})
    if num_layers >= 5:
        param_groups.append({'params': model.layer1.parameters(), 'lr': 1e-5})
    
    return optim.AdamW(param_groups, weight_decay=weight_decay)



def train_model(model, criterion, num_epochs,  epochs_per_stage, max_l):
    since = time.time()

  
    best_acc = 0.0
    best_model_path = f"best_multiclass_model_l.pt"
    torch.save(model.state_dict(), best_model_path)

    train_acc_list = []
    val_acc_list = []
    
    prev_update_step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 20)
        
          
        num_layers, cur_update_step = schedule(num_epochs, epoch, epochs_per_stage, max_l)
           
        if prev_update_step != cur_update_step:
            prev_update_step = cur_update_step
            set_finetune_l_layers(model, num_layers)
            #trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = build_optimizer(model, num_layers)
            
        print("Trainable parts:", describe_layers(num_layers))
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

    time_elapsed = time.time() - since
    print(f"\nTraining done in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    return model, train_acc_list, val_acc_list


def eval_test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs).argmax(1)     # ← model() per batch, same as training
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
    

def run(i):
    
    seed = 42 + i
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    all_results = {} # for plotting accuracy graph
    max_l = 5
    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 37)
    model = model.to(device)
    epochs_per_stage = 5
    epochs = epochs_per_stage * max_l
    
    #Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    
    #optimizer uses all trainable parameters (those not frozen)
    #trainable_params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.Adam(trainable_params, lr=0.001) 

    #Train
    model, train_acc, val_acc = train_model(
        model,
        criterion,
        epochs,
        epochs_per_stage,
        max_l
    )
    
    plt.figure()

    x = range(1, len(val_acc) + 1)
    plt.plot(x, train_acc, label=f"train", linestyle="--")
    plt.plot(x, val_acc, label=f"val")

    plt.title(f"Run {i} - Gradual unfreezing — train and val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f"compare_all_l_run_{i}.png")
    plt.close()
    
    test_acc = eval_test(model, dataloaders["test"], device )
    
    np.save(f"run_{i}_train.npy", np.array(train_acc))
    np.save(f"run_{i}_val.npy", np.array(val_acc))
    np.save(f"run_{i}_test.npy", np.array([test_acc]))
    return train_acc, val_acc, test_acc

def main():
    all_run_train = [] 
    all_run_val = []
    all_run_test = []
    
    max_run = 5
    for i in range(max_run):
        train, val, test = run(i)
        all_run_train.append(train)
        all_run_val.append(val)
        all_run_test.append(test)
        
    all_run_train = np.array(all_run_train)
    all_run_val = np.array(all_run_val)
    all_run_test = np.array(all_run_test)
    
    ave_train_acc = all_run_train.mean(axis=0)   
    ave_val_acc = all_run_val.mean(axis=0)  
    
    std_test_acc = all_run_test.std()
    ave_test_acc = all_run_test.mean() 
    
    print(f"Average train acc: {ave_train_acc}") 
    print(f"Average val acc: {ave_val_acc}") 
    print(f"Test accuracy: {ave_test_acc:.4f} ± {std_test_acc:.4f}")
        

if __name__ == "__main__":
    main()
    
    
        