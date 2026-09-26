from pathlib import Path

import torch
from torchvision.io import read_image, ImageReadMode

def compute_rgb_mean(image_dir):

    image_paths = sorted(Path(image_dir).glob("*.png"))

    channel_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for image_path in image_paths:
        image = read_image(image_path, ImageReadMode.RGB).to(torch.float64)

        channel_sum += image.sum(dim=(1, 2))
        total_pixels += image.shape[1] * image.shape[2]

    return channel_sum / total_pixels / 255.0