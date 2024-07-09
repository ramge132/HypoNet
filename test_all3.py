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

# root_dir = '/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14freeu_new_3'

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

def get_model(architecture, num_classes):
    if architecture == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif architecture == "inception_v3":
        # Inception_v3 requires the input size to be (299,299)
        model = models.inception_v3(weights=None, num_classes=num_classes, aux_logits=False)
    elif architecture == "densenet":
        model = models.densenet121(weights=None, num_classes=num_classes)
    elif architecture == "efficientnet":
        # Here we can choose from 'efficientnet_b0' to 'efficientnet_b7'
        model = models.efficientnet_b0(weights=None, num_classes=num_classes)
    elif architecture == "vit":
        # ViT requires the input size to be (224,224) or larger
        model = models.vit_b_16(weights=None, num_classes=num_classes)
    elif architecture == "swin":
        # Swin Transformer requires the input size to be (224,224) or larger
        model = models.swin_t(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def train_and_test(model_name, root_dir, resize_size, dataset_paths, classes, transform_train, transform_test):
    print(f"\nTraining {model_name} with images resized to {resize_size}x{resize_size}...")

    # 모델, 손실 함수, 최적화 함수 정의
    model = get_model(model_name, len(classes))

    # 모델별로 다른 optimizer 설정
    if model_name == "resnet18":
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    elif model_name == "inception_v3":
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    elif model_name == "densenet":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif model_name == "efficientnet":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    elif model_name in ["vit", "swin"]:
        optimizer = optim.AdamW(model.parameters(), lr=0.003 if model_name == "vit" else 0.0005)
        
    # 사용 가능한 GPU의 수에 따라 모델을 설정
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()

    # Early Stopping 관련 설정
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0

    # 데이터셋 및 데이터로더 재정의
    train_dataset = CustomCIFAR10Dataset(root_dir=root_dir, classes=classes, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)

    val_dataset = CustomCIFAR10Dataset(root_dir=dataset_paths[0], classes=classes, transform=transform_test)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)

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

        if epoch % 50 == 0:
            save_dir = f'saved_models'
            os.makedirs(save_dir, exist_ok=True)
            save_path = f'{save_dir}/{root_dir}_{model_name}_{resize_size}x{resize_size}_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
        
        # 검증 과정 
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)

        # Early Stopping 체크
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch} epochs!")
                break

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
    transforms.Resize((32, 32),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
])

transform_96 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
])

transform_224 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
])

transform_299 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299),interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
])

selected_classes = ['automobile', 'cat', 'horse', 'ship', 'truck']

# 각 모델에 대한 학습 및 테스트 실행
# resnet, densnet, efficientnet은 다양한 입력사이즈 가능
# inception은 299, vit, swin은 224x224 고정

# train_and_test(model_name, root_dir, resize_size, dataset_paths, classes, transform_train, transform_test):

# train_and_test('resnet18', root_dir, 32, [dataset_paths[0]], selected_classes, transform_32, transform_32_test)
# train_and_test('resnet18', root_dir, 96, [dataset_paths[1]], selected_classes, transform_96, transform_96_test)
# train_and_test('resnet18', root_dir, 224, [dataset_paths[2]], selected_classes, transform_224, transform_224_test)
train_and_test('inception_v3', root_dir, 299, [dataset_paths[0]], selected_classes, transform_299, transform_299)
train_and_test('inception_v3', root_dir, 299, [dataset_paths[1]], selected_classes, transform_299, transform_299)
train_and_test('inception_v3', root_dir, 299, [dataset_paths[2]], selected_classes, transform_299, transform_299)
train_and_test('densenet', root_dir, 224, [dataset_paths[0]], selected_classes, transform_224, transform_224)
train_and_test('densenet', root_dir, 224, [dataset_paths[1]], selected_classes, transform_224, transform_224)
train_and_test('densenet', root_dir, 224, [dataset_paths[2]], selected_classes, transform_224, transform_224)
train_and_test('efficientnet', root_dir, 224, [dataset_paths[0]], selected_classes, transform_224, transform_224)
train_and_test('efficientnet', root_dir, 224, [dataset_paths[1]], selected_classes, transform_224, transform_224)
train_and_test('efficientnet', root_dir, 224, [dataset_paths[2]], selected_classes, transform_224, transform_224)
train_and_test('vit', root_dir, 224, [dataset_paths[0]], selected_classes, transform_224, transform_224)
train_and_test('vit', root_dir, 224, [dataset_paths[1]], selected_classes, transform_224, transform_224)
train_and_test('vit', root_dir, 224, [dataset_paths[2]], selected_classes, transform_224, transform_224)
train_and_test('swin', root_dir, 224, [dataset_paths[0]], selected_classes, transform_224, transform_224)
train_and_test('swin', root_dir, 224, [dataset_paths[1]], selected_classes, transform_224, transform_224)
train_and_test('swin', root_dir, 224, [dataset_paths[2]], selected_classes, transform_224, transform_224)
