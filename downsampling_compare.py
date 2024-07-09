import cv2
import os
import numpy as np
from PIL import Image, ImageFilter

def main():
    # 이미지 로딩
    sample_image_path = '/media/taeyeong/T7 Shield/data/dataset/cifar10_custom_nextphoto_v30/airplane/162.png'  # 바꾸고자 하는 이미지
    img = cv2.imread(sample_image_path)

    # 다운샘플링 방법
    resampling_methods = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
        #"antialias": Image.LANCZOS
    }

    # 이미지 저장 디렉토리
    save_dir = '/media/taeyeong/T7 Shield/data/downsampled_samples'
    os.makedirs(save_dir, exist_ok=True)

    # 다운샘플링과 저장

    for method_name, interpolation in resampling_methods.items():
        if method_name == "antialias":
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            resized_img = pil_img.resize((32,32), Image.LANCZOS)
            resized_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_BGR2RGB)
        else:
            resized_img = cv2.resize(img, (32, 32), interpolation=interpolation)

        save_path = os.path.join(save_dir, f'sample_{method_name}.jpg')
        cv2.imwrite(save_path, resized_img)

if __name__ == "__main__":
    main()
