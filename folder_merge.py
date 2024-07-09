import os
import shutil
from tqdm import tqdm

source_dirs = [
    "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14_1",
    "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14_2"
]
dest_dir = "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14_3"

# 폴더 이름만 추출하여 출력합니다.
print("합치려는 폴더:")
for dir in source_dirs:
    print(f"- {os.path.basename(dir)}")

print("\n목적지 폴더:")
print(f"- {os.path.basename(dest_dir)}")

print("\n복사 작업을 시작합니다.")

categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 대상 디렉토리가 없으면 생성합니다.
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for category in tqdm(categories, desc='Categories'):
    category_dest_dir = os.path.join(dest_dir, category)
    # 각 카테고리의 대상 디렉토리가 없으면 생성합니다.
    if not os.path.exists(category_dest_dir):
        os.makedirs(category_dest_dir)

    for source_dir in source_dirs:
        category_source_dir = os.path.join(source_dir, category)
        # 각 카테고리의 소스 디렉토리가 존재하는 경우에만 작업을 수행합니다.
        if os.path.exists(category_source_dir):
            for filename in tqdm(os.listdir(category_source_dir), desc=f'Copying {category}', leave=False):
                source_file = os.path.join(category_source_dir, filename)
                dest_file = os.path.join(category_dest_dir, filename)

                # 파일명이 겹친다면 _1을 붙입니다.
                counter = 1
                while os.path.exists(dest_file):
                    name, ext = os.path.splitext(filename)
                    dest_file = os.path.join(category_dest_dir, f"{name}_{counter}{ext}")
                    counter += 1

                # 파일을 복사합니다.
                shutil.copy2(source_file, dest_file)

print("복사 작업이 완료되었습니다.")
