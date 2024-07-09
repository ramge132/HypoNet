import os
from cleanfid import fid

fdir1 = '/media/taeyeong/T7 Shield/data/downsampled_samples'
fdir2 = '/media/taeyeong/T7 Shield/data/dataset/cifar10_original/train'
#score = fid.compute_fid(fdir1, fdir2, mode="clean")
score = fid.compute_fid(fdir1, dataset_name="cifar10", dataset_res=32, mode="clean", dataset_split="train") # library의 cifar10 값으로 비교
print(f"FID score of {os.path.basename(fdir1)}: {score:.3f}")

# https://github.com/GaParmar/clean-fid

