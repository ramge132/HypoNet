import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from pytorch_fid import fid_score
from tqdm import tqdm
from scipy.stats import entropy

root_dir = '/media/taeyeong/T7 Shield/data/dataset/cifar10_xl_6'

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
        label = self.labels[idx]
        
        # 이미지 읽기
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 변환 적용 (예: 리사이징, 정규화 등)
        if self.transform:
            image = self.transform(image)

        return image, label

def torch_cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

# FID Calculation
def calculate_fid(model, real_loader, fake_loader, device):
    model.eval()
    real_acti = []
    fake_acti = []
    with torch.no_grad():
        for real_images, _ in tqdm(real_loader, desc="Calculating FID for real images"):
            real_images = real_images.to(device)
            act = model(real_images)
            
            # 각 배치에서의 출력을 리스트에 추가하기 전에 텐서를 재구성
            if isinstance(act, (list, tuple)):
                act = act[0]
            
            # 4차원 텐서를 2차원 텐서로 변환
            act = act.view(act.shape[0], -1)
            
            real_acti.append(act)
        
        for fake_images, _ in tqdm(fake_loader, desc="Calculating FID for fake images"):
            fake_images = fake_images.to(device)
            act = model(fake_images)
            
            # 각 배치에서의 출력을 리스트에 추가하기 전에 텐서를 재구성
            if isinstance(act, (list, tuple)):
                act = act[0]
            
            # 4차원 텐서를 2차원 텐서로 변환
            act = act.view(act.shape[0], -1)
            
            fake_acti.append(act)

    # 결과를 torch.Tensor로 연결
    real_acti = torch.cat(real_acti, dim=0)
    fake_acti = torch.cat(fake_acti, dim=0)
    
    mu1, sigma1 = real_acti.mean(0).cpu(), torch_cov(real_acti, rowvar=False).cpu()
    mu2, sigma2 = fake_acti.mean(0).cpu(), torch_cov(fake_acti, rowvar=False).cpu()

    
    fid = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def calculate_inception_score(model, loader, device, splits=10):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Calculating Inception Score"):
            images = images.to(device)
            output = model(images)
            if isinstance(output, (list, tuple)):  # 이 부분을 추가
                output = output[0]
            # Softmax를 취하여 확률 분포를 얻음
            probs = torch.nn.functional.softmax(output, dim=1)
            preds.append(probs.cpu().numpy())
    
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# Extract features using InceptionV3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = fid_score.InceptionV3().to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for InceptionV3")
    inception_model = nn.DataParallel(inception_model)

# 1. custom 32x32
custom_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32), interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet
    #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # CIFAR10
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1 to 1
])
custom_dataset = CustomCIFAR10Dataset(root_dir=root_dir, transform=custom_transform)
custom_loader = DataLoader(custom_dataset, batch_size=512, shuffle=False, num_workers=32)

# 2. custom 299x299
custom_transform_299 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299), interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet
    #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # CIFAR10
])
custom_dataset_299 = CustomCIFAR10Dataset(root_dir=root_dir, transform=custom_transform_299)
custom_loader_299 = DataLoader(custom_dataset_299, batch_size=512, shuffle=False, num_workers=32) # 배치 크기를 줄임

# 3. cifar original
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet
    #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # CIFAR10
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1 to 1
])
cifar_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
cifar_train_loader = DataLoader(cifar_train_dataset, batch_size=512, shuffle=False, num_workers=32)

# 4. cifar origianl 299
cifar_transform_299 = transforms.Compose([
    transforms.Resize((299, 299), interpolation=cv2.INTER_CUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet의 평균 및 표준 편차를 사용한 정규화
    #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # CIFAR10의 평균 및 표준 편차를 사용한 정규화
])
cifar_train_dataset_299 = CIFAR10(root='./data', train=True, download=True, transform=cifar_transform_299)
cifar_train_loader_299 = DataLoader(cifar_train_dataset_299, batch_size=512, shuffle=False, num_workers=32)



# Calculate FID between CIFAR-10 training images and your custom images
fid_value = calculate_fid(inception_model, cifar_train_loader_299, custom_loader_299, device)
print(f"root dir: {root_dir}")
print(f"FID between CIFAR-10 training images and custom images: {fid_value}")

# Inception Score 계산
#is_mean, is_std = calculate_inception_score(inception_model, custom_loader_299, device)
#print(f"Inception Score for custom images: {is_mean} ± {is_std}")