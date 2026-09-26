from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from datasets.sr_transform import SRTrainTransform, SRValidTransform, SRTestTransform, subtract_rgb_mean


class SRDataset(Dataset):

    def __init__(self, lr_path, hr_path, split, scale, lr_patch_size, rgb_mean):
        super().__init__()

        lr_dir = Path(lr_path)
        hr_dir = Path(hr_path)

        self.rgb_mean = rgb_mean

        lr_files = {path.name : path for path in lr_dir.glob("*.png")}
        hr_files = {path.name : path for path in hr_dir.glob("*.png")}

        self.pairs = [(lr_files[name], hr_files[name]) for name in sorted(lr_files)]

        if split == "train":
            self.transform = SRTrainTransform(scale, lr_patch_size)
        elif split == "valid":
            self.transform = SRValidTransform()
        elif split == "test":
            self.transform = SRTestTransform()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        lr_path, hr_path = self.pairs[index]

        lr = read_image(lr_path, ImageReadMode.RGB)
        hr = read_image(hr_path, ImageReadMode.RGB)

        lr, hr = self.transform(lr, hr)

        lr = subtract_rgb_mean(lr, self.rgb_mean)
        hr = subtract_rgb_mean(hr, self.rgb_mean)

        return lr, hr

