import os
import shutil
import random

# 원본 디렉토리 설정
#source_root = '/media/taeyeong/T7 Shield/data/dataset/cifar10_original/train'
source_root = '/media/taeyeong/T7 Shield/data/dataset/cifar10_sd15freeu_new_3'

# 대상 디렉토리 설정
#dest_root = '/home/taeyeong/Documents/images'
dest_root = '/media/taeyeong/T7 Shield/data/dataset/cifar10_sd15freeu_new_4'

# 클래스 목록
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 각 클래스별로 복사할 파일의 개수
#num_files_to_copy = 40
num_files_to_copy = 5000

# 대상 디렉토리가 없으면 생성
if not os.path.exists(dest_root):
    os.makedirs(dest_root)

# 각 클래스에 대해 반복
for cls in classes:
    source_dir = os.path.join(source_root, cls)
    dest_dir = os.path.join(dest_root, cls)

    # 각 클래스의 대상 디렉토리가 없으면 생성
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 원본 디렉토리에서 모든 파일 목록을 가져옴
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 랜덤하게 'num_files_to_copy'개의 파일을 선택
    selected_files = random.sample(all_files, min(num_files_to_copy, len(all_files)))

    # 선택된 파일을 대상 디렉토리로 복사
    for file in selected_files:
        shutil.copy2(os.path.join(source_dir, file), os.path.join(dest_dir, file))

print("파일 복사 작업이 완료되었습니다.")
