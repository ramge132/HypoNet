import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import timm

# 1. 데이터 전처리 및 DataLoader 설정
transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    #transforms.CenterCrop(444),
    transforms.Resize(224),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='/media/taeyeong/T7 Shield/data/dataset/cifar10_sdlora_3', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=15)

# 모델 선택 및 설정
model_names = {  # <- timm에서 사용하는 모델 이름 지정
    "vitl": "vit_large_patch16_224.augreg_in21k",
    "vitb": "vit_base_patch16_224.augreg_in21k",
    "res": "resnet18",
    "eff0": "tf_efficientnet_b0_ns"
}

selected_model = "res"
if selected_model in model_names:
    model = timm.create_model(model_names[selected_model], pretrained=False, num_classes=15)
else:
    raise ValueError(f"Model {selected_model} not available in timm.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
    
# 최적화 알고리즘 설정
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 학습 및 테스트 함수 정의
def train(epoch):
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} - Training Loss: {total_loss / len(train_loader)}")

def test(test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    return 100.0 * correct / len(test_loader.dataset)

# 학습 및 테스트 실행
for epoch in range(1, 101):
    train(epoch)
    if epoch in [50, 100]:
        torch.save(model.module.state_dict(), f"model_{model.__class__.__name__}_{epoch}.pth")
        
        # CIFAR-10 테스트
        cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        cifar_test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False, num_workers=4)
        print(f"Epoch {epoch} - CIFAR-10 Accuracy: {test(cifar_test_loader)}%")
        
        # # STL-10 테스트
        # # 'car'를 'automobile'로 변경하고, 필요한 클래스만 선택
        # desired_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']
        # class_to_idx = {cls: i for i, cls in enumerate(desired_classes)}
        # idx_to_class = {v: k for k, v in class_to_idx.items()}

        # stl_test = datasets.STL10(root='./data', split='test', download=True, transform=transform)
        # stl_labels = [label for _, label in stl_test]
        # filtered_indices = [i for i, label in enumerate(stl_labels) if stl_test.classes[label] in desired_classes]
        # stl_test = Subset(stl_test, filtered_indices)

        # # DataLoader에 STL-10 데이터셋을 로드하면서 'car' 레이블을 'automobile'로 변경
        # def collate_fn(batch):
        #     images, labels = zip(*batch)
        #     labels = [class_to_idx.get(stl_test.dataset.classes[label], label) if stl_test.dataset.classes[label] == 'car' else label for label in labels]
        #     return torch.stack(images, 0), torch.tensor(labels)

        # stl_test_loader = DataLoader(stl_test, batch_size=64, shuffle=False, num_workers=15, collate_fn=collate_fn)

        # # 모델의 출력 특성을 9로 변경합니다.
        # model.module.head = torch.nn.Linear(model.module.head.in_features, 9).cuda()
        # print(f"Epoch {epoch} - STL-10 Accuracy: {test(stl_test_loader)}%")

        # # 모델의 출력 특성을 다시 10으로 변경하여 다음 테스트를 준비합니다.
        # model.module.head = torch.nn.Linear(model.module.head.in_features, 10).cuda()
        
        # ImageNet 검증 데이터셋 테스트
        imagenet_class_ids = {
            'airplane': ['n04552348'],
            'automobile': ['n03100240', 'n04285008'],
            'bird': ['n01518878','n01582220'],
            'cat': ['n02123045', 'n02123394'],
            'deer': ['n02422699', 'n02423022'],
            'dog': ['n02085620', 'n02086240'],
            'frog': ['n01641577','n01644373'],
            'horse': ['n02389026'],
            'ship': ['n03095699', 'n04612504'],
            'truck': ['n03417042', 'n04467665']
        }
        valid_ids = sum(imagenet_class_ids.values(), [])
        imagenet_val = datasets.ImageFolder(root='/media/taeyeong/T7 Shield/data/dataset/imagenet_val', transform=transform)
        valid_indices = [i for i, (_, target) in enumerate(imagenet_val) if imagenet_val.classes[target] in valid_ids]
        imagenet_val_filtered = Subset(imagenet_val, valid_indices)
        imagenet_val_loader = DataLoader(imagenet_val_filtered, batch_size=64, shuffle=False, num_workers=15)
        print(f"Epoch {epoch} - ImageNet Validation Accuracy: {test(imagenet_val_loader)}%")

print("학습 종료")
