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
        RandomHorizontalFlip(),  # Randomly flip the image horizontally
        #RandomCrop(32, padding=4),  # Randomly crop the image and pad if needed
        transforms.RandomRotation(15),  # Rotate by up to 10 degrees
        #transforms.Resize((224, 224), interpolation=interpolation),  
        transforms.ToTensor()  
        #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    dataset.transform = transform

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=15)

    #-------------------------Architectures-------------------------#
    
    # ResNet-18
    model, model_name = models.resnet18(weights=None, num_classes=10), "ResNet-18"
    
    # ResNet-34
    #model, model_name = models.resnet34(weights=None, num_classes=10), "ResNet-34"
    
    # VGG 16
    #model, model_name = models.vgg16(weights=None, num_classes=10), "VGG16"

    # EfficientNet B0
    #model, model_name = timm.create_model('efficientnet_b0', weights=None, num_classes=10), "EfficientNet B0"

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
        if epoch % 50 == 0:
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
for method, model in trained_models.items():  # modified to unpack both method and model
    #for epoch in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    for epoch in [50, 100]:
        print(f"Testing with {method} downsampling at epoch {epoch}...")

        model_path = f'./saved_models/{model_name}_{method}_{epoch}.pth'
        
        if model_name == "ResNet-18":
            model = models.resnet18(weights=None, num_classes=10)  # corrected 'weights' to 'pretrained'
        elif model_name == "ResNet-34":
            model = models.resnet34(weights=None, num_classes=10)  # corrected 'weights' to 'pretrained'
        elif model_name == "VGG16":
            model = models.vgg16(weights=None, num_classes=10)  # corrected 'weights' to 'pretrained'
        elif model_name == "EfficientNet B0":
            model = timm.create_model('efficientnet_b0', weights=None, num_classes=10)
        elif model_name == "DenseNet-121":
            model = models.densenet121(weights=None, num_classes=10)  # corrected 'weights' to 'pretrained'
        else:
            raise ValueError("Unknown model name")

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(model_path))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        class_true_positive = list(0. for i in range(10))
        class_false_positive = list(0. for i in range(10))
        class_false_negative = list(0. for i in range(10))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # 전체 정밀도 및 재현율을 계산하기 위한 변수 초기화
        total_true_positive = 0
        total_false_positive = 0
        total_false_negative = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    pred = predicted[i]
                    class_total[label] += 1  # 실제 레이블에 대한 전체 수를 업데이트합니다.
                    if pred == label:
                        class_true_positive[label] += 1
                        total_true_positive += 1  # 전체 true positive 업데이트
                    else:
                        class_false_negative[label] += 1  # 실제 클래스에 대한 false negative 업데이트
                        class_false_positive[pred] += 1  # 잘못 예측된 클래스에 대한 false positive 업데이트
                        total_false_negative += 1  # 전체 false negative 업데이트
                        total_false_positive += 1  # 전체 false positive 업데이트

        total_accuracy = 100 * sum(class_true_positive) / sum(class_total)
        print(f'Accuracy of the model trained with {method} downsampling on the test images: {total_accuracy}%')

        writer.add_scalar(f'{method}/Test Accuracy at epoch {epoch}', total_accuracy, epoch)

        # 전체 정밀도 및 재현율 계산
        total_precision = (total_true_positive / (total_true_positive + total_false_positive)) if (total_true_positive + total_false_positive) != 0 else 0
        total_recall = (total_true_positive / (total_true_positive + total_false_negative)) if (total_true_positive + total_false_negative) != 0 else 0

        print(f'모델의 전체 정밀도: {total_precision * 100:.2f}%')
        print(f'모델의 전체 재현율: {total_recall * 100:.2f}%')

        for i in range(10):
            precision = (class_true_positive[i] / (class_true_positive[i] + class_false_positive[i])) if (class_true_positive[i] + class_false_positive[i]) != 0 else 0
            recall = (class_true_positive[i] / (class_true_positive[i] + class_false_negative[i])) if (class_true_positive[i] + class_false_negative[i]) != 0 else 0
            print(f'For {class_names[i]} - Precision: {100 * precision:.2f}%, Recall: {100 * recall:.2f}%')

writer.close()
