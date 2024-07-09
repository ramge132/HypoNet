#!/bin/bash

src="/media/taeyeong/T7 Shield/data/dataset/cifar10_xl_"
dst="/media/taeyeong/T7 Shield/data/dataset/cifar10_xlfreeu_2"

for src_file in "$src"/*; do
  base_file=$(basename "$src_file")
  dst_file="$dst/$base_file"
  
  counter=1
  while [[ -e "$dst_file" ]]; do
    dst_file="$dst/${base_file%.*}_$counter.${base_file##*.}"
    ((counter++))
  done
  
  mv "$src_file" "$dst_file"
done
