#!/bin/bash

# 원본 이미지가 있는 디렉토리
SOURCE_DIRECTORY="/media/taeyeong/T7 Shield/data/outputs2/txt2img-images/2023-11-07"

# 이미지를 저장할 디렉토리 
DEST_DIRECTORY="/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14freeu_new_1"

# 원본 디렉토리가 존재하는지 확인
if [ ! -d "$SOURCE_DIRECTORY" ]; then
    echo "Error: Source directory $SOURCE_DIRECTORY does not exist."
    exit 1
fi

# 목적지 디렉토리가 존재하는지 확인, 없으면 생성
if [ ! -d "$DEST_DIRECTORY" ]; then
    echo "Destination directory $DEST_DIRECTORY does not exist."
    echo "Creating destination directory."
    mkdir -p "$DEST_DIRECTORY"
fi

# 파일 이름 변경
count=95623 # 시작 숫자
for img in "$SOURCE_DIRECTORY"/*.png; do
    mv "$img" "$DEST_DIRECTORY/$count.png"
    count=$((count + 1))
done

echo "Renaming completed."
