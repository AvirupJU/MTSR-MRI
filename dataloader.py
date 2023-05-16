import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, src_dir, tgt_dir):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.list_files = os.listdir(self.src_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        src_path = os.path.join(self.src_dir, img_file)
        tgt_path = os.path.join(self.tgt_dir, img_file)
        src_img = np.array(Image.open(src_path))[:,240:480,:3]
        tgt_img = np.array(Image.open(src_path))[:,240:480,:3]

        #augmentations = config.both_transform(image=input_image, image0=target_image)
        #input_image = augmentations["image"]
        #target_image = augmentations["image0"]

        input_image = config.lowres_transform_A(image=src_img)["image"]
        target_image = config.highres_transform(image=tgt_img)["image"]
        #target_image = tgt_img

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("/home/adey/enhancement/datasets/BRATS/brats_18/train/","/home/adey/enhancement/datasets/BRATS/brats_18/train/")
    loader = DataLoader(dataset, batch_size=8)
    #print(dataset.shape)
    for x, y in loader:
        print(x.shape, y.shape)
        save_image(x, "x.jpg")
        save_image(y, "y.jpg")
        import sys

        sys.exit()