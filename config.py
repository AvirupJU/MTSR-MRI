from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


TRAIN_DIR = "/home/adey/enhancement/datasets/BRATS/brats_18/train/"
VAL_DIR = "/home/adey/enhancement/datasets/BRATS/brats_18/val/"
CHECKPOINT_GEN = "/home/adey/enhancement/Real-ESRGAN/weights/rrdb_t2.pth.tar"
CHECKPOINT_DISC = "/home/adey/enhancement/Real-ESRGAN/weights/disc_t2.pth.tar"

LOAD_GMODEL = True
LOAD_DMODEL = False
SAVE_MODEL = True


NUM_EPOCHS = 91
PREPOCHS = 10
BATCH_SIZE = 12
LEARNING_RATE = 2e-4
NUM_WORKERS = 4
LAMBDA_L1 = 2 
LAMBDA_VGG = 200
LOW_RES = 64
HIGH_RES = 256

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lowres_transform_A = A.Compose(
    [
        #first order
        A.OneOf([
            A.Resize(height=LOW_RES, width=LOW_RES, interpolation=cv2.INTER_AREA),
            A.Resize(height=LOW_RES, width=LOW_RES, interpolation=cv2.INTER_LINEAR),
            A.Resize(height=LOW_RES, width=LOW_RES, interpolation=cv2.INTER_CUBIC),
        ], p=1),

        A.OneOf([
           A.GaussNoise(),
           A.ISONoise(),
           A.ColorJitter(), 
        ], p=1),

        #A.ImageCompression(quality_lower=75, quality_upper=85, p=1),

        #second order degradation
        #A.OneOf([
        #    A.Resize(height=LOW_RES, width=LOW_RES, interpolation=cv2.INTER_AREA),
        #    A.Resize(height=LOW_RES, width=LOW_RES, interpolation=cv2.INTER_LINEAR),
        #    A.Resize(height=LOW_RES, width=LOW_RES, interpolation=cv2.INTER_CUBIC),
        #], p=0.1),
#
        #A.OneOf([
        #   A.GaussNoise(),
        #   A.ISONoise(),
        #   A.ColorJitter(), 
        #], p=0.75),


        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2(),
    ]
)

highres_transform = A.Compose(
    [
        A.Resize(height=HIGH_RES, width=HIGH_RES, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2(),
    ]
)
