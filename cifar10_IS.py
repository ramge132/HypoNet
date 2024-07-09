import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# 사용자 정의 전처리 함수 정의
def preprocess_input_custom(image):
    image = image - 0.5
    image *= 2.0
    return image

class CIFAR10Dataset(Dataset):
    def __init__(self, folder_path, image_size=(299, 299)):  
        self.transform = transforms.Compose([
            #transforms.Resize(image_size),
            transforms.Resize((32, 32), interpolation=Image.BICUBIC),  # 먼저 이미지를 32x32로 리사이즈 bicubic
            transforms.Resize(image_size, interpolation=Image.BICUBIC), 
            transforms.ToTensor(),
            transforms.Lambda(preprocess_input_custom), 
        ])

        self.images = []
        for class_folder in os.listdir(folder_path):
            class_folder_path = os.path.join(folder_path, class_folder)
            for img_name in os.listdir(class_folder_path):
                if img_name.endswith('.png') or img_name.endswith('.jpg'):
                    img_path = os.path.join(class_folder_path, img_name)
                    self.images.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img

def inception_score(img_loader, cuda=True, batch_size=32, splits=10, eps=1E-16):
    N = len(img_loader.dataset)

    assert batch_size > 0
    assert N > batch_size

    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).eval() 
    if cuda:
        inception_model = inception_model.cuda()

    preds = []
    for i, batch in enumerate(tqdm(img_loader, desc="Inception Score 계산 중")):
        if cuda:
            batch = batch.cuda()

        with torch.no_grad():
            pred = inception_model(batch)
            preds.append(torch.nn.functional.softmax(pred, dim=-1).data.cpu().numpy())

    preds = np.concatenate(preds, 0)

    scores = []
    for i in range(splits):
        part = preds[(i * N // splits):((i + 1) * N // splits), :]
        kl = part * (np.log(part + eps) - np.log(np.expand_dims(np.mean(part, 0) + eps, 0))) 
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)

data_dir = "/media/taeyeong/T7 Shield/data/dataset/cifar10_sdxlfreeu_new_3" 
dataset = CIFAR10Dataset(data_dir)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=15) 

mean, std = inception_score(data_loader, cuda=True, batch_size=32, splits=10)
print("Inception Score: {:.2f} ± {:.2f}".format(mean, std))
