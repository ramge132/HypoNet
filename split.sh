#!/bin/bash

# 원본 디렉토리
SOURCE_DIRECTORY="/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14freeu_new_1"

# # 대상 디렉토리 및 파일 범위 설정
# declare -A DEST_DIRECTORIES=(
#     ["airplane"]="1 5000"
#     ["automobile"]="5001 10000"
#     ["bird"]="10001 15000"
#     ["cat"]="15001 20000"
#     ["deer"]="20001 25000"
#     ["dog"]="25001 30000"
#     ["frog"]="30001 35000"
#     ["horse"]="35001 40000"
#     ["ship"]="40001 45000"
#     ["truck"]="45001 50000"
# )

# 대상 디렉토리 및 파일 범위 설정
declare -A DEST_DIRECTORIES=(
    ["airplane"]="1 10000"
    ["automobile"]="10001 20000"
    ["bird"]="20001 30000"
    ["cat"]="30001 40000"
    ["deer"]="40001 50000"
    ["dog"]="50001 60000"
    ["frog"]="60001 70000"
    ["horse"]="70001 80000"
    ["ship"]="80001 90000"
    ["truck"]="90001 100000"
)

# 각 대상 디렉토리에 대해 파일 이동 수행
for category in "${!DEST_DIRECTORIES[@]}"; do
    start=$(echo ${DEST_DIRECTORIES[$category]} | cut -d' ' -f1)
    end=$(echo ${DEST_DIRECTORIES[$category]} | cut -d' ' -f2)
    dest="$SOURCE_DIRECTORY/$category"

    # 대상 디렉토리가 없으면 생성
    if [ ! -d "$dest" ]; then
        mkdir -p "$dest"
    fi

    # 파일 이동
    for i in $(seq -f "%g.png" $start $end); do
        mv "$SOURCE_DIRECTORY/$i" "$dest/"
    done
done

echo "Files have been moved."
