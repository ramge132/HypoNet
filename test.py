import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


root_dir = '/media/taeyeong/T7 Shield/data/dataset/cifar10_sdnone_1'
custom_dataset_paths = [
    '/media/taeyeong/T7 Shield/data/dataset/cifar10_original/test',
    '/media/taeyeong/T7 Shield/data/dataset/imagenet_val',
    '/media/taeyeong/T7 Shield/data/dataset/stl10_test',
]

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
    "cubic": cv2.INTER_CUBIC,
}

# Load the data once to use memory efficiently
dataset = CustomCIFAR10Dataset(root_dir=root_dir)

# Train model with various downsampling methods
for method, interpolation in resampling_methods.items():
    print(f"Training with {method} downsampling...")
    
    # Adding data augmentation techniques
    transform = transforms.Compose([
        transforms.ToPILImage(),
        RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(15),  # Rotate by up to 10 degrees
        #transforms.CenterCrop(363), # if training image is 512x512
        #transforms.CenterCrop(725), # if training image is 1024x1024
        transforms.Resize((32, 32), interpolation=interpolation),  
        transforms.ToTensor()  
    ])
    
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=24)
    
    # ResNet-18
    model = models.resnet18(weights=None, num_classes=10)
    model_name = "ResNet-18"
    print(f"Architecture: {model_name}") 
    print(f"root_dir: {root_dir}")
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(1, 51):
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

        if epoch % 50 == 0:
            os.makedirs('./saved_models', exist_ok=True)
            save_path = f'./saved_models/{model_name}_{method}_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")

    trained_model = model
    trained_model_name = f'{model_name}_{method}'

# Testing phase
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

for dataset_path in custom_dataset_paths:
    dataset_name = os.path.basename(dataset_path)
    print(f"\nTesting {trained_model_name} on {dataset_name} dataset...")

    custom_dataset = CustomCIFAR10Dataset(root_dir=dataset_path, transform=test_transform)
    custom_loader = DataLoader(custom_dataset, batch_size=64, shuffle=False)

    model_path = f'./saved_models/{trained_model_name}_50.pth'
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        for images, labels in custom_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    total_accuracy = 100 * sum(class_correct) / sum(class_total)
    print(f'Accuracy of the model on the {dataset_name} dataset: {total_accuracy:.2f}%')
    writer.add_scalar(f'{trained_model_name}/Test Accuracy on {dataset_name} Dataset', total_accuracy)

    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of {class_names[i]} : {100 * class_correct[i] / class_total[i]:.2f} %')
        else:
            print(f'Accuracy of {class_names[i]} : N/A (no samples)')

writer.close()
