# ------------- 5개의 클래스에 대해서만 학습 및 테스트 하는 코드 ----------------- #

import os
import cv2
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser(description='Run tests on different datasets')
parser.add_argument('--root_dir', type=str, required=True, help='Root directory for the dataset')
args = parser.parse_args()
root_dir = args.root_dir

#root_dir = '/media/taeyeong/T7 Shield/data/dataset/cifar10_sdxllora_3'

dataset_paths = [
    '/media/taeyeong/T7 Shield/data/dataset/cifar10_original/test',
    '/media/taeyeong/T7 Shield/data/dataset/imagenet_val',
    '/media/taeyeong/T7 Shield/data/dataset/stl10_test',
]

# Custom CIFAR-10 dataset class
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
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

    # CV
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

    # # PIL
    # def __getitem__(self, idx):
    #     img_path = self.data[idx]
    #     img = Image.open(img_path).convert("RGB")

    #     if self.transform:
    #         img = self.transform(img)

    #     label = self.labels[idx]
    #     return img, label

# # Custom dataset에 대해 Normalize 
# dataset = CustomCIFAR10Dataset(root_dir=root_dir, transform=transforms.ToTensor())
# loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)

# # 평균 및 표준편차 계산
# def compute_mean_std(loader):
#     mean = 0.
#     std = 0.
#     nb_samples = 0.
#     for data, _ in tqdm(loader):
#         batch_samples = data.size(0)
#         data = data.view(batch_samples, data.size(1), -1)
#         mean += data.mean(2).sum(0)
#         std += data.std(2).sum(0)
#         nb_samples += batch_samples
#     mean /= nb_samples
#     std /= nb_samples
#     return mean, std

# mean, std = compute_mean_std(loader)
# print(f'Mean: {mean}, Std: {std}')

def train_and_test(model_name, root_dir, resize_size, dataset_paths, classes, transform_train, transform_test):
    print(f"\nTraining {model_name} with images resized to {resize_size}x{resize_size}...")

    # 모델, 손실 함수, 최적화 함수 정의
    model = models.resnet18(weights=None, num_classes=len(classes))
    
    # 사용 가능한 GPU의 수에 따라 모델을 설정
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 데이터셋 및 데이터로더 재정의
    train_dataset = CustomCIFAR10Dataset(root_dir=root_dir, classes=classes, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)

    # 학습
    for epoch in range(1, 51):
        model.train()
        epoch_loss = 0.0
        epoch_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            epoch_loss += loss.item() * inputs.size(0)
            epoch_corrects += torch.sum(preds == labels.data)

        print('Training Loss:', epoch_loss / len(train_loader.dataset))
        print('Training Accuracy:', epoch_corrects.double() / len(train_loader.dataset))

        if epoch % 50 == 0:
            save_dir = f'saved_models'
            os.makedirs(save_dir, exist_ok=True)
            save_path = f'{save_dir}/{model_name}_{resize_size}x{resize_size}_{epoch}.pth'
            torch.save(model.state_dict(), save_path)

    # 테스트
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f"\nTesting model trained on '{os.path.basename(root_dir)}' using {dataset_name} dataset...")

        test_dataset = CustomCIFAR10Dataset(root_dir=dataset_path, classes=classes, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16)

        corrects = 0
        total = 0
        corrects_per_class = [0] * 10
        total_per_class = [0] * 10

        with torch.no_grad():
            model.eval()
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # 각 클래스별 정확도 업데이트
                for label, pred in zip(labels, preds):
                    if label == pred:
                        corrects_per_class[label.item()] += 1
                    total_per_class[label.item()] += 1
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        accuracy = 100 * corrects / total
        print(f'Average Accuracy on {dataset_name} dataset: {accuracy:.2f} %')

        # 각 클래스별 정확도 출력
        for i, class_name in enumerate(test_dataset.classes):
            class_accuracy = 100 * corrects_per_class[i] / total_per_class[i]
            print(f"Accuracy for class {class_name}: {class_accuracy:.2f} %")

        # 'stl10_test' 데이터셋에 대해서만 'frog' 클래스를 제외하고 평균 정확도 계산
        if dataset_name == 'stl10_test':
            sum_accuracy = 0
            count = 0
            for i, class_name in enumerate(test_dataset.classes):
                if class_name != 'frog':
                    class_accuracy = corrects_per_class[i] / total_per_class[i]
                    sum_accuracy += class_accuracy
                    count += 1
            avg_accuracy_excluding_frog = 100 * sum_accuracy / count
            print(f'Average Accuracy Excluding Frog on STL10 Test Dataset: {avg_accuracy_excluding_frog:.2f} %')


# CV 학습 및 테스트를 위한 변환 정의
transform_32 = transforms.Compose([
    transforms.ToPILImage(),
    #RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    transforms.Resize((32, 32),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std),
])

transform_96 = transforms.Compose([
    transforms.ToPILImage(),
    #RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    transforms.Resize((96, 96),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std),
])

transform_224 = transforms.Compose([
    transforms.ToPILImage(),
    #RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    transforms.Resize((224, 224),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std),
])

transform_32_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std),
])

transform_96_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std),
])

transform_224_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std),
])


# # PIL 학습 및 테스트를 위한 변환 정의
# transform_32 = transforms.Compose([
#     #RandomHorizontalFlip(),
#     #transforms.RandomRotation(15),
#     #transforms.CenterCrop(363), # if training image is 512x512
#     #transforms.CenterCrop(725), # if training image is 1024x1024
#     transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
# ])

# transform_96 = transforms.Compose([
#     #RandomHorizontalFlip(),
#     #transforms.RandomRotation(15),
#     #transforms.CenterCrop(363), # if training image is 512x512
#     #transforms.CenterCrop(725), # if training image is 1024x1024
#     transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
# ])

# transform_224 = transforms.Compose([
#     #RandomHorizontalFlip(),
#     #transforms.RandomRotation(15),
#     #transforms.CenterCrop(363), # if training image is 512x512
#     #transforms.CenterCrop(725), # if training image is 1024x1024
#     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
# ])

# transform_32_test = transforms.Compose([
#     transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
# ])

# transform_96_test = transforms.Compose([
#     transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
# ])

# transform_224_test = transforms.Compose([
#     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
# ])

selected_classes = ['automobile', 'cat', 'horse', 'ship', 'truck']

# 각 모델에 대한 학습 및 테스트 실행
train_and_test('ResNet-18', root_dir, 32, [dataset_paths[0]], selected_classes, transform_32, transform_32_test)
#train_and_test('ResNet-18', root_dir, 96, [dataset_paths[1]], selected_classes, transform_96, transform_96_test)
#train_and_test('ResNet-18', root_dir, 224, [dataset_paths[2]], selected_classes, transform_224, transform_224_test)