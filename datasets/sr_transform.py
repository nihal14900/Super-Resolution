import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

class SRTrainTransform:

    def __init__(self, scale, lr_patch_size):

        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.to_float = v2.ToDtype(torch.float32, True)

    def __call__(self, lr, hr):

        _, height, width = lr.shape

        top  = torch.randint(0, height - self.lr_patch_size + 1, ()).item()
        left = torch.randint(0, width - self.lr_patch_size + 1, ()).item()

        lr = F.crop(lr, top, left, self.lr_patch_size, self.lr_patch_size)
        hr = F.crop(hr, self.scale * top, self.scale * left, self.scale * self.lr_patch_size, self.scale * self.lr_patch_size)

        if torch.rand(()).item() < 0.5:
            lr = F.horizontal_flip(lr)
            hr = F.horizontal_flip(hr)

        if torch.rand(()).item() < 0.5:
            lr = F.vertical_flip(lr)
            hr = F.vertical_flip(hr)

        k = torch.randint(0, 4, ()).item()

        lr = torch.rot90(lr, k, (-2, -1))
        hr = torch.rot90(hr, k, (-2, -1))

        lr = self.to_float(lr)
        hr = self.to_float(hr)

        lr = lr.contiguous()
        hr = hr.contiguous()

        return lr, hr

class SRValidTransform:

    def __init__(self):

        self.to_float = v2.ToDtype(torch.float32, True)
    
    def __call__(self, lr, hr):

        lr = self.to_float(lr)
        hr = self.to_float(hr)

        lr = lr.contiguous()
        hr = hr.contiguous()

        return lr, hr

class SRTestTransform:

    def __init__(self):
    
        self.to_float = v2.ToDtype(torch.float32, True)

    def __call__(self, lr, hr):

        lr = self.to_float(lr)
        hr = self.to_float(hr)

        lr = lr.contiguous()
        hr = hr.contiguous()

        return lr, hr

def subtract_rgb_mean(image, rgb_mean):
    mean = rgb_mean.to(device = image.device, dtype = image.dtype).view(3, 1, 1)

    return image - mean

def add_rgb_mean(image, rgb_mean):
    mean = rgb_mean.to(device = image.device, dtype = image.dtype).view(1, 3, 1, 1)

    return image + mean