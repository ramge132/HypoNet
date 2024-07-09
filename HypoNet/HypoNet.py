"""
════════════════════════════════════════
                                 HypoNet
           Image Classification and Hyponym Extraction with Deep Learning
════════════════════════════════════════
Author: Yu Tae Yeong
Date: 2023
 
This software is part of the HypoNet project. It is designed to classify images
and extract relevant hyponyms using deep learning and natural language processing.

COPYRIGHT © 2023 Yu Tae Yeong. All Rights Reserved.

Contact: taeyoun9@gmail.com
════════════════════════════════════════
"""

import torch
from torchvision import models, transforms
from PIL import Image
from nltk.corpus import wordnet as wn
import json
import random
from torchvision.models import ResNet50_Weights

# 이미지를 분류하는 함수
def predict_image_class(image_path):
    # 모델 불러오기
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()  # 평가 모드로 설정

    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 이미지 로드 및 전처리
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # 예측된 클래스 인덱스 가져오기
    _, pred_idx = torch.max(output, 1)
    pred_idx = pred_idx.item()  # 인덱스를 integer로 변환

    return pred_idx

# 이미지의 가장 구체적인 클래스에서 상위 개념을 찾는 함수
def find_hypernyms(class_index, class_index_json):
    with open(class_index_json) as json_file:
        class_idx_to_label = json.load(json_file)
    # 가장 구체적인 클래스 레이블 찾기
    specific_class = class_idx_to_label[str(class_index)][1]
    
    # 해당 클래스의 모든 synset 찾기
    synsets = wn.synsets(specific_class, pos=wn.NOUN)
    hypernym_sets = set()
    # 각 synset에 대해 상위 개념 찾기
    for synset in synsets:
        for hypernym in synset.hypernyms():
            hypernym_sets.add(hypernym)
    
    # 상위 개념 중 하나를 선택 (예: 첫 번째)
    if not hypernym_sets:
        return None
    hypernym = list(hypernym_sets)[0]
    return hypernym

# WordNet을 사용하여 하위 분류 조회
def get_hyponyms(synset, max_count):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= {lemma.name().replace('_', ' ') for lemma in hyponym.lemmas()}
    hyponyms = list(hyponyms)
    
    # 랜덤하게 하위 분류 선택
    if max_count < len(hyponyms):
        hyponyms = random.sample(hyponyms, max_count)
    
    return hyponyms

# 결과를 파일로 저장
def save_to_file(filename, hyponyms):
    with open(filename, 'w') as f:
        for hyponym in hyponyms:
            f.write(f"{hyponym}\n")

# 메인 함수
def main(image_path, max_count, class_index_json):
    # 이미지 분류
    class_index = predict_image_class(image_path)
    # 상위 개념 찾기
    hypernym_synset = find_hypernyms(class_index, class_index_json)
    if hypernym_synset:
        # 하위 분류 조회
        hyponyms = get_hyponyms(hypernym_synset, max_count)
        # 파일 이름 설정
        filename = f"{hypernym_synset.lemmas()[0].name()}.txt"
        # 파일로 저장
        save_to_file(filename, hyponyms)
        print(f"Hyponyms for '{hypernym_synset.lemmas()[0].name()}' are saved to '{filename}'.")
    else:
        print("Could not find a hypernym for the predicted class.")

# 사용 예:
main('/mnt/4TB/graduation/selective_prompt/deer.JPEG', 10, 'imagenet_class_index.json')
