import os
import cv2
from PIL import Image

def resize_save_images(method, original_dir, save_dir):
    # 하위 디렉토리 목록 가져오기
    sub_dirs = [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))]

    for sub_dir in sub_dirs:
        # 원본 하위 디렉토리 경로
        original_sub_dir = os.path.join(original_dir, sub_dir)
        # 저장 하위 디렉토리 경로
        save_sub_dir = os.path.join(save_dir, sub_dir)
        
        # 저장 하위 디렉토리 생성
        if not os.path.exists(save_sub_dir):
            os.makedirs(save_sub_dir)
        
        # 이미지 파일 목록 가져오기
        images = [f for f in os.listdir(original_sub_dir) if os.path.isfile(os.path.join(original_sub_dir, f))]
        for image_file in images:
            # 이미지 열기
            if method == 'PIL':
                with Image.open(os.path.join(original_sub_dir, image_file)) as img:
                    # 이미지 다운샘플링
                    img_resized = img.resize((32, 32), Image.BICUBIC)
                    # 이미지 저장
                    img_resized.save(os.path.join(save_sub_dir, image_file))
            elif method == 'OpenCV':
                img = cv2.imread(os.path.join(original_sub_dir, image_file))
                # 이미지 다운샘플링
                img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
                # 이미지 저장
                cv2.imwrite(os.path.join(save_sub_dir, image_file), img_resized)

# 원본 경로 지정
original_dir = '/media/taeyeong/T7 Shield/data/dataset/cifar10_sdxlfreeu_new_3'

# PIL 및 OpenCV로 이미지 크기 조정 및 저장
resize_save_images('PIL', original_dir, '/media/taeyeong/T7 Shield/data/downsampled_samples')  # PIL 저장 경로를 여기에 지정하세요.
#resize_save_images('OpenCV', original_dir, '/mnt/4TB/graduation/downsampled_dataset/cifar10_sdlora_3_cv')  # OpenCV 저장 경로를 여기에 지정하세요.

print("모든 이미지의 크기 조정 및 저장이 완료되었습니다.")
