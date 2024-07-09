import os
import shutil
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from cifar10_models.densenet import densenet161
#from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

# 1. 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, src_dir, transform=None):
        self.src_dir = src_dir
        self.transform = transform
        self.image_paths = glob.glob(f"{src_dir}/*/*")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_path

# Initialize TensorBoard writer
#writer = SummaryWriter('runs/image_classification')

# Initialize device and model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = densenet161(pretrained=True)
model.to(device)  

# Use DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
    print(f"GPUs being used: {model.device_ids}")

model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Source and destination directories
src_dir = "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14freeu_new_1"
dst_dir = "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14freeu_new_2"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Threshold for classification
threshold = 0.95

# Create dataset and dataloader
dataset = CustomDataset(src_dir=src_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=8)

# Process images in batches using the dataloader
for batch_imgs, batch_img_paths in tqdm(dataloader):
    batch_imgs = batch_imgs.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch_imgs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        max_probs, predicted_classes = torch.max(probs, 1)
        
    # Check if the images belong to the correct classes and exceed the threshold
    for i in range(len(batch_imgs)):
        img_path = batch_img_paths[i]
        predicted_class = predicted_classes[i]
        max_prob = max_probs[i]
        class_name = os.path.basename(os.path.dirname(img_path))
        
        if class_names[predicted_class] == class_name and max_prob >= threshold:
            dst_class_dir = os.path.join(dst_dir, class_name)
            
            # 해당 디렉토리가 존재하는지 확인하고, 존재하지 않으면 생성합니다.
            if not os.path.exists(dst_class_dir):
                os.makedirs(dst_class_dir)
            
            dst_img_path = os.path.join(dst_class_dir, os.path.basename(img_path))
            shutil.copy(img_path, dst_img_path)
            
            # Log to TensorBoard
            unnormalized_img_tensor = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
            #writer.add_image(f"{class_name}/{os.path.basename(img_path)}", unnormalized_img_tensor, global_step=0)

# Close TensorBoard writer
#writer.close()
