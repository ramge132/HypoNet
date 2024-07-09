# 1. resampling_methods
# 2. root_dir
# 3. augmentations
# 4. architectures
# 5. optimizers & lr
# 6. epochs

import os
import cv2
import time
import timm
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

# Define various downsampling methods
resampling_methods = {
    #"nearest": cv2.INTER_NEAREST,
    #"linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    #"lanczos4": cv2.INTER_LANCZOS4,
    #"area": cv2.INTER_AREA
}

# Load the data once to use memory efficiently
root_dir = '/media/taeyeong/T7 Shield/data/dataset/cifar10_lora_1'
dataset = CustomCIFAR10Dataset(root_dir=root_dir)

# Dictionary to store trained models for each downsampling method
trained_models = {}

# Train models with various downsampling methods
for method, interpolation in resampling_methods.items():
    print(f"Training with {method} downsampling...")
    
    # Adding data augmentation techniques
    transform = transforms.Compose([
        transforms.ToPILImage(),
        #RandomHorizontalFlip(),  # Randomly flip the image horizontally
        #RandomCrop(32, padding=4),  # Randomly crop the image and pad if needed
        #transforms.RandomRotation(15),  # Rotate by up to 10 degrees
        transforms.Resize((32, 32), interpolation=interpolation),  
        transforms.ToTensor()  
    ])
    
    dataset.transform = transform

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=24)

    #-------------------------Architectures-------------------------#
    
    # ResNet-18
    model, model_name = models.resnet18(weights=None, num_classes=10), "ResNet-18"
    
    # ResNet-34
    #model, model_name = models.resnet34(weights=None, num_classes=10), "ResNet-34"
    
    # VGG 16
    #model, model_name = models.vgg16(weights=None, num_classes=10), "VGG16"

    # EfficientNet B0
    #model, model_name = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10), "EfficientNet B0"

    # DenseNet 121
    #model, model_name = models.densenet121(weights=None, num_classes=10), "DenseNet-121"

    #--------------------------------------------------------------#

    print(f"Architecture: {model_name}") 
    print(f"root_dir: {root_dir}")
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)


    start_time = time.time()
    #for epoch in range(1, 1001):
    for epoch in range(1, 101):
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

        writer.add_scalar(f'{method}/Training Loss', epoch_loss / len(loader.dataset), epoch)
        writer.add_scalar(f'{method}/Training Accuracy', epoch_corrects.double() / len(loader.dataset), epoch)

        #if epoch % 100 == 0:
        if epoch % 20 == 0:
            os.makedirs('./saved_models', exist_ok=True)
            save_path = f'./saved_models/{model_name}_{method}_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")

    trained_models[method] = model

# Load test dataset and set DataLoader
test_transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the performance of trained models
for method, _ in trained_models.items():
    #for epoch in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    for epoch in [20, 40, 60, 80, 100]:
        print(f"Testing with {method} downsampling at epoch {epoch}...")

        model_path = f'./saved_models/{model_name}_{method}_{epoch}.pth'
        
        if model_name == "ResNet-18":
            model = models.resnet18(weights=None, num_classes=10)
        elif model_name == "ResNet-34":
            model = models.resnet34(weights=None, num_classes=10)            
        elif model_name == "VGG16":
            model = models.vgg16(weights=None, num_classes=10)            
        elif model_name == "EfficientNet B0":
            model = timm.create_model('efficientnet_b0', weights=None, num_classes=10)
        elif model_name == "DenseNet-121":
            model = models.densenet121(weights=None, num_classes=10)
        else:
            raise ValueError("Unknown model name")

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
        print(f'Accuracy of the model trained with {method} downsampling on the test images: {total_accuracy}%')

        writer.add_scalar(f'{method}/Test Accuracy at epoch {epoch}', total_accuracy, epoch)

        for i in range(10):
            print(f'Accuracy of {class_names[i]} : {100 * class_correct[i] / class_total[i]} %')

writer.close()
