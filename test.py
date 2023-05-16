import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure
import os
import numpy as np
import config 
from PIL import Image
from torchvision.utils import save_image
from utils import save_checkpoint, load_checkpoint
from slim_gen import generator
import time

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


t1_ckpt_file = '/home/adey/enhancement/Real-ESRGAN/weights/rrdb_t1.pth.tar'
t2_ckpt_file = '/home/adey/enhancement/Real-ESRGAN/weights/rrdb_t2.pth.tar'
folder = '/home/adey/enhancement/Real-ESRGAN/test_inputs'
save_folder = '/home/adey/enhancement/Real-ESRGAN/test_output_t1'

gen = generator.model
gen = nn.DataParallel(gen)
checkpoint = torch.load(t2_ckpt_file, map_location=config.DEVICE)
gen.load_state_dict(checkpoint["state_dict"])
#x = torch.randn((1,3,64,64))
#out = gen(x)
#print(out.shape)
bicubic = nn.Upsample(scale_factor=4, mode='bicubic')
bilinear = nn.Upsample(scale_factor=4, mode='bilinear')
snr_metric = PSNR()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

def compute_metrics(x, y, snr_metric, ssim_metric):
    ssim = ssim_metric(x, y)
    snr = snr_metric(torch.round((x + 1) * 255 / 2), torch.round((y + 1) * 255 / 2))
    return [snr, ssim]

def upscale(in_img, gen):
    out_img = gen(in_img.unsqueeze(0).to(config.DEVICE))
    return out_img

ssim, psnr = 0,0
t0 = time.time()
for Img in os.listdir(folder):
    N = len(os.listdir(folder))
    img = Image.open(folder+'/'+Img)
    img = np.array(img)[:,240:480,:3] #0:240 for t1, 240:480 for t2
    lr_img = config.lowres_transform_A(image=img)["image"]
    hr_img = config.highres_transform(image=img)["image"]
    hr_img = hr_img.unsqueeze(0)
    #out = upscale(lr_img, gen).to('cpu')
    #bicubic_out = bicubic(lr_img.unsqueeze(0))
    bilinear_out = bilinear(lr_img.unsqueeze(0))
    #save_image(torch.concat([bilinear_out, bicubic_out, out, hr_img, hr_img]), f"{save_folder}/{Img}")
    #psnr += compute_metrics(out, hr_img, snr_metric, ssim_metric)[0]
    #ssim += compute_metrics(out, hr_img, snr_metric, ssim_metric)[1]
t1 = time.time()
print(t1-t0)    
#print(f"mean ssim: {ssim/N} \nmean psnr: {psnr/N}")
