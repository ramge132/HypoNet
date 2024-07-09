from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image
import gradio as gr
import json
import random
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from nltk.corpus import wordnet as wn
import os
os.environ['CURL_CA_BUNDLE'] = ''

# Hugging Face 토큰
HUGGINGFACE_TOKEN = "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"

# 이미지 분류 및 하이포님 추출
def predict_image_class(image_data):
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

    # 이미지 데이터가 PIL 이미지 객체인지 확인
    if isinstance(image_data, Image.Image):
        input_image = image_data
    else:
        # 이미지가 numpy 배열로 제공되면 PIL 이미지로 변환합니다.
        input_image = Image.fromarray(image_data)
    
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
    pred_idx = pred_idx.item()  # 인덱스를 정수로 변환

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

# 이미지 파일과 최대 하위 분류 갯수를 받아서 결과를 반환하는 함수
def classify_and_find_hyponyms(image, max_count):
    max_count = int(max_count)  # max_count를 정수로 변환
    # 이미지 분류
    class_index = predict_image_class(image)
    # 상위 개념 찾기
    hypernym_synset = find_hypernyms(class_index, 'imagenet_class_index.json')
    if hypernym_synset:
        # 하위 분류 조회
        hyponyms = get_hyponyms(hypernym_synset, max_count)
        # 결과 반환
        return '\n'.join(hyponyms)
    else:
        return "Could not find a hypernym for the predicted class."
    

# Stable Diffusion 이미지 생성 함수
def setup_stable_diffusion_model(model_choice):
    # 모델 선택에 따라서 적절한 모델을 설정합니다.
    if model_choice == "Stable Diffusion v1-4":
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
    elif model_choice == "Stable Diffusion v1-5":
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
    elif model_choice == "Stable Diffusion 2-1 Base":
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", revision="fp16", torch_dtype=torch.float16)
    else:
        raise ValueError(f"Invalid model choice: {model_choice}")

    # GPU 사용 설정
    pipe = pipe.to("cuda")
    return pipe

def generate_image(pipe, hyponym):
    prompt = f"(best quality, 8k, realistic, photo-realistic:1.4), {hyponym}"
    negative_prompt = "(worst quality:1.4, low quality:1.4, normal quality:1.4, watermark:1.4, nsfw:1.4), drawings, abstract art, cartoons, surrealist painting, conceptual drawing, graphics, bad proportions, human, person, people, man, woman, girl, boy"
    
    # sampling_step과 CFG scale 값을 설정합니다.
    generation_kwargs = {
        "num_inference_steps": 20,  # sampling steps
        "guidance_scale": 7,       # CFG scale
    }
    
    with torch.no_grad():
        # 생성된 이미지를 반환합니다.
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=512, width=512, **generation_kwargs).images[0]
    return image

# Gradio Blocks 인터페이스와 통합
with gr.Blocks() as demo:
    gr.Markdown("# HypoNet & Stable Diffusion *(Yu Tae Yeong, 2023)*")  # This will add a title at the top of the interface

    with gr.Row():
        image = gr.Image(label="Upload Image", type="pil")
        max_count = gr.Number(label="Max Hyponyms Count", default=10)
        model_choice = gr.Radio(choices=["Stable Diffusion v1-4", "Stable Diffusion v1-5", "Stable Diffusion 2-1 Base"], label="Model Choice")
        generate_btn = gr.Button("Generate with HypoNet").style(background_color="#FFA500", color="white")

    
    hyponyms_str = gr.Textbox(label="Hyponyms List")
    generated_images = gr.Gallery(label="Generated Images").style(grid=[2, 5])

    def classify_and_generate(image, max_count, model_choice):
        # 이미지 분류 및 하이포님 추출
        hyponyms_str_value = classify_and_find_hyponyms(image, max_count)
        if not hyponyms_str_value.startswith("Could not find"):
            # 이미지 생성
            pipe = setup_stable_diffusion_model(model_choice)
            hyponyms = hyponyms_str_value.split('\n')
            generated_images = [generate_image(pipe, hyponym) for hyponym in hyponyms]
            return hyponyms_str_value, generated_images
        else:
            # 하이포님을 찾을 수 없는 경우, 빈 이미지 목록 반환
            return hyponyms_str_value, []
    

    generate_btn.click(
        classify_and_generate,
        inputs=[image, max_count, model_choice],
        outputs=[hyponyms_str, generated_images]
    )

    with gr.Row():
        reload_btn = gr.Button("Reload UI")

    def reset_interface():
        # Clear all inputs and outputs
        image.update(None)
        max_count.update(10)
        model_choice.update("Stable Diffusion v1-4")
        hyponyms_str.update("")
        generated_images.update([])

        # You can also reset any internal state here if necessary
        # For example, if you have any global variables or a session state, reset them here

        return "", [], []  # Return empty values to clear the interface

    reload_btn.click(
        reset_interface,
        inputs=[],
        outputs=[hyponyms_str, generated_images]
    )

demo.launch(server_name='114.70.21.211', server_port=7863)