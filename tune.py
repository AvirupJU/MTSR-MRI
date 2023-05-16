from PIL import Image
import albumentations as A
import cv2
import numpy as np
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from dataloader import MapDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from discriminator import Discriminator
from slim_gen import generator
from vgg_loss import VGGLoss

torch.backends.cudnn.benchmark = True

def pretrain_generator(net_G, train_dl, l1_loss, opt, epochs):
    for ep in range(epochs):
        for x, y in tqdm(train_dl):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            preds = net_G(x)
            loss = l1_loss(preds, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        print(f"Epoch {ep + 1}/{epochs}")
        print(f"L1 Loss: {loss.mean().item():.5f}")
        save_checkpoint(net_G, opt, filename=config.CHECKPOINT_GEN)


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, vgg_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        x_u = F.interpolate(x, scale_factor=4, mode='nearest')
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x_u, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x_u, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x_u, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.LAMBDA_L1
            VGG = vgg_loss(y_fake, y) * config.LAMBDA_VGG
            G_loss = (1/200) * (G_fake_loss + L1 + VGG)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = generator.model
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    VGG_LOSS = VGGLoss()

    train_dataset = MapDataset(src_dir=config.TRAIN_DIR, tgt_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(src_dir=config.VAL_DIR, tgt_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    print("pretraining genrator:")
    pretrain_generator(gen, train_loader, L1_LOSS, opt_gen, config.PREPOCHS)

    if config.LOAD_GMODEL:
        #gen = nn.DataParallel(gen)
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
    if config.LOAD_DMODEL:
        #disc = nn.DataParallel(disc)
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    
    "start gan training:"
    gen = nn.DataParallel(gen)
    disc = nn.DataParallel(disc)
    for epoch in range(config.NUM_EPOCHS):
        print(f"{epoch}/{config.NUM_EPOCHS}")
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, VGG_LOSS, BCE, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

            save_some_examples(gen, val_loader, epoch, folder="evaluation_t1")


if __name__ == "__main__":
    main()

#print(net.model)

#path_to_image = "/home/adey/enhancement/datasets/new_maps/val/src/68.jpg"
#image = Image.open(path_to_image).convert('RGB')
#image = np.array(image)
#input_image = lowres_transform_A(image=image)["image"]
#inp = Image.fromarray(input_image)
#inp.save('inputs/68.jpg')
#
#sr_image = net.predict(input_image)
#
#sr_image.save('results/68_slim_odd.jpg')
#print("done!")

#print(gen.model.body)