import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os
import numpy as np
import config 
from PIL import Image
#from torchvision.utils import save_image
#from utils import save_checkpoint, load_checkpoint
#from slim_gen import generator
import timeit

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
all_realism_scores = []
all_diversity_scores = []

root = "/home/adey/enhancement/BicycleGAN/results_3/val/images"

SAMPLES = ["000","001","002","003","004","005","006","007","008","009","010","011","012"]
#img_1_path = f"{root}/input_000_ground truth.png"
'''
for i in range(1,5):
    img_1_path = f"{root}/input_" + sample + f"_random_sample0{i}.png"
    img_2_path = f"{root}/input_" + sample + f"_random_sample0{i+1}.png"


    img_1 = np.array(Image.open(img_1_path))
    img_2 = np.array(Image.open(img_2_path))

    img_1 = config.highres_transform(image=img_1)["image"]
    img_2 = config.highres_transform(image=img_2)["image"]
    img_1 = img_1.unsqueeze(0)
    img_2 = img_2.unsqueeze(0)
    img_1 = img_1 * 2 - 1
    img_2 = img_2 * 2 - 1
    #print(img_1.shape, img_2.shape)
    #print(lpips(img_1, img_2).detach())
    diversity_scores.append(lpips(img_1, img_2).item())
print(np.mean(diversity_scores))
'''
for sample in SAMPLES:
    realism_scores = []
    for i in range(5):
        gt_path = f"{root}/input_" + sample + "_ground truth.png"
        img_path = f"{root}/input_" + sample + f"_random_sample0{i+1}.png"

        gt = np.array(Image.open(gt_path))
        img = np.array(Image.open(img_path))

        gt = config.highres_transform(image=gt)["image"]
        img = config.highres_transform(image=img)["image"]

        gt = gt.unsqueeze(0)
        img = img.unsqueeze(0)
        gt = gt * 2 - 1
        img = img * 2 - 1

        realism_scores.append(lpips(gt, img).item())

    all_realism_scores.append(np.mean(realism_scores))

print(np.mean(all_realism_scores))