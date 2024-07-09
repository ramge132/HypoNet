#!/bin/bash

# Python 버전 확인
python_version=$(python3 --version 2>&1)
echo -e "1. 파이썬 버전: $python_version\n"

# PyTorch 버전 확인
torch_version=$(python3 -c "import torch; print(torch.__version__)")
echo -e "2. 파이토치 버전: $torch_version\n"

# TorchVision 버전 확인
torchvision_version=$(python3 -c "import torchvision; print(torchvision.__version__)")
echo -e "3. 토치비전 버전: $torchvision_version\n"

# CUDA 버전 확인 (시스템)
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo -e "4. 시스템 쿠다 버전: $cuda_version\n"

# PyTorch가 사용하는 CUDA 버전 확인
pytorch_cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
echo -e "5. 파이토치 쿠다 버전: $pytorch_cuda_version\n"

# cuDNN 버전 확인
cudnn_version=$(cat /usr/local/cuda/include/cudnn_version.h | grep "#define CUDNN_MAJOR" -A 2 | awk '{print $3}' | xargs echo | sed 's/ /./g')
echo -e "6. cuDNN 버전: $cudnn_version\n"