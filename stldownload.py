import os
import torchvision.datasets as datasets
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# STL-10 데이터셋의 클래스 이름
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# 저장할 디렉토리 경로
save_dir = '/media/taeyeong/T7 Shield/data/dataset/stl10_test'

# STL-10 데이터셋 다운로드 (데이터셋이 이미 다운로드 되어있다면, download=False로 설정)
stl10_data = datasets.STL10(root='.', split='test', download=True) # split = 'train', 'test', 'unlabeled' 

# 클래스별 디렉토리 생성
for cls in classes:
    os.makedirs(os.path.join(save_dir, cls), exist_ok=True)

# STL-10 데이터셋의 이미지를 클래스별 디렉토리에 PNG 파일로 저장
for i, (image, label) in enumerate(tqdm(stl10_data)):
    class_name = classes[label]
    image.save(os.path.join(save_dir, class_name, f'image_{i}.png'))
