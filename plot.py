import matplotlib.pyplot as plt
import pandas as pd

# 데이터 구조화
data = {
    "Trainset": [
        "HypoNet (SD XL)", "LoRA (SD XL)", "FreeU (SD XL)", "- (SD XL)",
        "HypoNet (SD 1.5)", "LoRA (SD 1.5)", "FreeU (SD 1.5)", "- (SD 1.5)",
        "HypoNet (SD 1.4)", "LoRA (SD 1.4)", "FreeU (SD 1.4)", "- (SD 1.4)",
        "CIFAR-10 Train"
    ],
    "Accuracy Average": [68.03, 58.31, 53.08, 62.95, 71.05, 55.39, 62.60, 69.49, 69.04, 50.99, 61.40, 68.31, 70.50]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 바 그래프 그리기
plt.figure(figsize=(10, 6))
plt.barh(df['Trainset'], df['Accuracy Average'], color='skyblue')
plt.xlabel('Accuracy Average')
plt.ylabel('Trainset')
plt.title('Accuracy Average by Trainset')
plt.grid(axis='x')
plt.tight_layout()

# 그래프 출력
plt.show()
