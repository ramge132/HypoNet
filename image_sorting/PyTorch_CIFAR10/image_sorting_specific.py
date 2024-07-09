import os
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from cifar10_models.densenet import densenet161
from tensorboardX import SummaryWriter

# Initialize TensorBoard writer
writer = SummaryWriter('runs/image_classification')

# Initialize device and model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Remove this line
model = densenet161(pretrained=True)

# Use DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to("cuda")  # This will load the model to all available GPUs
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Source and destination directories
src_dir = "/media/taeyeong/T7 Shield/data/dataset/cifar10_sdfreeu_temp"
dst_dir = "/media/taeyeong/T7 Shield/data/dataset/cifar10_sdfreeu_temp2"

# Create destination directory if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)



#특정 클래스만 하고 싶다면
# Original CIFAR-10 class names
original_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Target class names
target_class_names = ['bird', 'cat', 'dog']

# Threshold for classification
threshold = 0.95 

# Loop through each class folder
for class_name in tqdm(original_class_names):
    if class_name not in target_class_names:
        continue

    src_class_dir = os.path.join(src_dir, class_name)
    dst_class_dir = os.path.join(dst_dir, class_name)
    
    if not os.path.exists(dst_class_dir):
        os.makedirs(dst_class_dir)
    
    for filename in os.listdir(src_class_dir):
        img_path = os.path.join(src_class_dir, filename)
        
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted_class = torch.max(probs, 1)
            
        # Check if the image belongs to the same class and exceeds the threshold
        if original_class_names[predicted_class] == class_name and max_prob >= threshold:
            dst_img_path = os.path.join(dst_class_dir, filename)
            shutil.copy(img_path, dst_img_path)
            
            # Log to TensorBoard
            unnormalized_img_tensor = transforms.ToTensor()(img) 
            writer.add_image(f"{class_name}/{filename}", unnormalized_img_tensor, global_step=0)



# # CIFAR-10 class names
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# # Threshold for classification
# threshold = 0.95

# # Loop through each class folder
# for class_name in tqdm(class_names):
#     src_class_dir = os.path.join(src_dir, class_name)
#     dst_class_dir = os.path.join(dst_dir, class_name)
    
#     if not os.path.exists(dst_class_dir):
#         os.makedirs(dst_class_dir)
    
#     for filename in os.listdir(src_class_dir):
#         img_path = os.path.join(src_class_dir, filename)
        
#         # Load and preprocess image
#         img = Image.open(img_path).convert('RGB')
#         img_tensor = transform(img).unsqueeze(0).to(device)
        
#         # Forward pass
#         with torch.no_grad():
#             outputs = model(img_tensor)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             max_prob, predicted_class = torch.max(probs, 1)
            
#         # Check if the image belongs to the same class and exceeds the threshold
#         if class_names[predicted_class] == class_name and max_prob >= threshold:
#             dst_img_path = os.path.join(dst_class_dir, filename)
#             shutil.copy(img_path, dst_img_path)
            
#             # Log to TensorBoard
#             unnormalized_img_tensor = transforms.ToTensor()(img)
#             writer.add_image(f"{class_name}/{filename}", unnormalized_img_tensor, global_step=0)
#             #writer.add_image(f"{class_name}/{filename}", img_tensor[0], global_step=0)

# Close TensorBoard writer
writer.close()
