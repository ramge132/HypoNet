import os
from pathlib import Path
import shutil
from tqdm import tqdm

def copy_images(src_dir, dst_dir, num_images, copy_all=False):
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for category in categories:
        src_path = Path(src_dir) / category
        dst_path = Path(dst_dir) / category
        dst_path.mkdir(parents=True, exist_ok=True)

        if copy_all:
            images = list(src_path.glob('*.png'))
        else:
            images = list(src_path.glob('*.png'))[:num_images]

        for img in tqdm(images, desc=f"Copying {category}"):
            dst_file = dst_path / img.name
            counter = 1

            # 파일 이름이 중복되면 이름 변경
            while dst_file.exists():
                name, ext = os.path.splitext(img.name)
                new_name = f"{name}_{counter}{ext}"
                dst_file = dst_path / new_name
                counter += 1

            shutil.copy(img, dst_file)

def main():
    base_src_dir = '/media/taeyeong/T7 Shield/data/dataset'
    base_dst_dir = '/media/taeyeong/T7 Shield/data/mixed_dataset'
    
    for percent in [20, 40, 60, 80, 100]:
        num_images = percent * 1000 // 20
        dst_dir = Path(base_dst_dir) / f'{percent}per'

        # Copy all images (5000) from train folder
        train_src_dir = Path(base_src_dir) / 'cifar10_original/train'
        copy_images(train_src_dir, dst_dir, num_images=5000, copy_all=True)

        # Copy a specified number of images from cifar10_sd_9 folder
        sd_src_dir = Path(base_src_dir) / 'cifar10_sd_9'
        copy_images(sd_src_dir, dst_dir, num_images)

if __name__ == '__main__':
    main()
