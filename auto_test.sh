#!/bin/bash

# root_dir 리스트
declare -a root_dirs=(
#"/media/taeyeong/T7 Shield/data/dataset/cifar10_original/train"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14none_1"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14freeu_new_3"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14lora_3" 
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd14_5"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sdnone_1" 
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd15freeu_new_4"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sdlora_3"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_sd_9"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_xlnone_2"
#"/media/taeyeong/T7 Shield/data/dataset/cifar10_sdxlfreeu_new_3"
# "/media/taeyeong/T7 Shield/data/dataset/cifar10_xl_6"
"/media/taeyeong/T7 Shield/data/mixed_dataset/20per"
"/media/taeyeong/T7 Shield/data/mixed_dataset/40per"
"/media/taeyeong/T7 Shield/data/mixed_dataset/60per"
"/media/taeyeong/T7 Shield/data/mixed_dataset/80per"
"/media/taeyeong/T7 Shield/data/mixed_dataset/100per"
)

# 로그 파일을 저장할 디렉토리
log_dir="/media/taeyeong/T7 Shield/data/logs"

# 로그 디렉토리가 없다면 생성
[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

# 각 root_dir에 대해 스크립트 실행
for root_dir in "${root_dirs[@]}"; do
  # 현재 날짜와 시간을 가져옴
  current_time=$(date "+%Y.%m.%d-%H.%M.%S")
  
  # 로그 파일 이름 설정
  log_file="$log_dir/log_$current_time.txt"
  
  # 파이썬 스크립트 실행 및 로그 파일에 출력 저장
  python3 test_all2.py --root_dir "$root_dir" | tee "$log_file"
done
