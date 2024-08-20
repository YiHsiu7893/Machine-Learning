import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm

from LoadData import CustomDataset
from LabelSmoothing import LabelSmoothingLoss


### Hyperparameters ###
re_size = 256
crop_size = 224
batch_size = 64
lr_begin = (batch_size / 256) * 0.1  # learning rate at begining
epochs = 60
seed = 0

# Set random seeds
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

np.random.seed(seed)


### Define Transforms ###
train_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.RandomCrop(crop_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def custom_collate_train(batch):
    images, labels = zip(*batch)
    images = [train_transform(img) for img in images]
    return torch.stack(images), torch.tensor(labels)
def custom_collate_val(batch):
    images, labels = zip(*batch)
    images = [val_transform(img) for img in images]
    return torch.stack(images), torch.tensor(labels)


### Load Datasets ###
dataset = CustomDataset('./data/train')

# Split into training and validation
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_train)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_val)


### Model Setting ###
net = resnet50(pretrained=True)
net.fc = torch.nn.Linear(net.fc.in_features, 200)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
#print(device)

for param in net.parameters():
    param.requires_grad = True

criterion = LabelSmoothingLoss(classes=200, smoothing=0.1)  # label smoothing to improve performance
optimizer = torch.optim.SGD(net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Check if a pre-trained model checkpoint exists
pretrained_model_path = './pretrained_model.pth'
if os.path.isfile(pretrained_model_path):
    # Load the model checkpoint
    checkpoint = torch.load(pretrained_model_path)

    # Load the model and optimizer states
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded pre-trained model.")
else:
    print(f"Training from scratch.")


### Training ###
previous_losses = './train_losses.npy'
previous_val_losses = './val_losses.npy'
previous_accs = './train_accs.npy'
previous_val_accs = './val_accs.npy'

if os.path.exists(previous_losses):
    train_losses = np.load(previous_losses)
    val_losses = np.load(previous_val_losses)
    train_accs = np.load(previous_accs)
    val_accs = np.load(previous_val_accs)
    print("Load previous Info.")
else:
    train_losses = np.array([])
    val_losses = np.array([])
    train_accs = np.array([])
    val_accs = np.array([])

# Training loop
for epoch in range(epochs):
    print('\n===== Epoch: {} ====='.format(epoch))
    net.train()

    train_loss = train_correct = train_total = idx = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx

        optimizer.zero_grad()  # Sets the gradients to zero
        inputs, targets = inputs.to(device), targets.to(device)

        y = net(inputs)
        loss = criterion(y, targets)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(y.data, 1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets.data).cpu().sum()
        train_loss += loss.item()

    scheduler.step()

    train_acc = 100.0 * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    train_accs = np.append(train_accs, train_acc)
    train_losses = np.append(train_losses, train_loss)

    print('Train | Loss: {:.4f} | Acc: {:.3f}% ({}/{})'.format(train_loss, train_acc, train_correct, train_total))


    ### Validation ###
    net.eval()  # Set model to evaluation mode

    val_loss = val_correct = val_total = idx = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, ncols=80)):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)

            y = net(inputs)
            loss = criterion(y, targets)

            _, predicted = torch.max(y.data, 1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets.data).cpu().sum()
            val_loss += loss.item()

    val_acc = 100.0 * float(val_correct) / val_total
    val_loss = val_loss / (idx + 1)
    val_accs = np.append(val_accs, val_acc)
    val_losses = np.append(val_losses, val_loss)

    print('Validation | Loss: {:.4f} | Acc: {:.3f}% ({}/{})'.format(val_loss, val_acc, val_correct, val_total))


    ### Save Info. ###
    torch.save({
      'model_state_dict': net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()},
      pretrained_model_path)
    np.save(previous_losses, train_losses)
    np.save(previous_val_losses, val_losses)
    np.save(previous_accs, train_accs)
    np.save(previous_val_accs, val_accs)