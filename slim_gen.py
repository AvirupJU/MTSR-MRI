import torch
from torch import nn
from realesrgan import RealESRGAN
from config import DEVICE

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


generator = RealESRGAN(DEVICE, scale=4)
generator.load_weights('weights/RealESRGAN_x4.pth')

for i in range(23):
    if i%2!=0:
        generator.model.body[i] = Identity()

def test ():
    x = torch.randn((1, 3, 64, 64)).to(DEVICE)
    y = generator.model(x)
    print(generator.model.body)
    print(x.shape, y.shape)

if __name__ == '__main__':
    test()
