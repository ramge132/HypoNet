import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.datasets import CIFAR10

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Custom CIFAR-10 dataset class
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.data = []
        self.labels = []

        for label, cls in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, cls)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

# Initialize TensorBoard writer
writer = SummaryWriter()

# Load the data once to use memory efficiently
root_dir = '/media/taeyeong/T7 Shield/data/dataset/cifar10_original/train'
dataset = CustomCIFAR10Dataset(root_dir=root_dir)

# Adding data augmentation techniques
transform = transforms.Compose([
    transforms.ToPILImage(),
    RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()  
])

dataset.transform = transform
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=24)

# Train the model
print(f"Training with original CIFAR-10 data...")

# ResNet-18
model, model_name = models.resnet18(weights=None, num_classes=10), "ResNet-18"

print(f"Architecture: {model_name}") 
print(f"root_dir: {root_dir}")
model = nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
start_time = time.time()
for epoch in range(1, 51):  # Changed to 50 epochs
    epoch_start_time = time.time()
    epoch_loss = 0.0
    epoch_corrects = 0

    tqdm_loader = tqdm(loader, desc=f"Epoch {epoch}")
    for i, (inputs, labels) in enumerate(tqdm_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)

        tqdm_loader.set_description(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    writer.add_scalar(f'Training Loss', epoch_loss / len(loader.dataset), epoch)
    writer.add_scalar(f'Training Accuracy', epoch_corrects.double() / len(loader.dataset), epoch)

    if epoch % 10 == 0:  # Save every 10 epochs
        os.makedirs('./saved_models', exist_ok=True)
        save_path = f'./saved_models/{model_name}_{epoch}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

# Evaluation
test_transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Test loop
for epoch in [10, 20, 30, 40, 50]:
    print(f"Testing the trained model at epoch {epoch}...")

    model_path = f'./saved_models/{model_name}_{epoch}.pth'
    model = models.resnet18(weights=None, num_classes=10)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_path))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    total_accuracy = 100 * sum(class_correct) / sum(class_total)
    print(f'Accuracy of the trained model on the test images: {total_accuracy}%')
    writer.add_scalar(f'Test Accuracy', total_accuracy, epoch)

    for i in range(10):
        print(f'Accuracy of {class_names[i]} : {100 * class_correct[i] / class_total[i]} %')

writer.close()
