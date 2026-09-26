import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

def rgb_to_y(image):
    r = image[:, 0:1]
    g = image[:, 1:2]
    b = image[:, 2:3]

    return 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0

@torch.no_grad()
def calculate_metrics(sr, hr, rgb_mean, scale):

    mean = rgb_mean.to(device = sr.device, dtype = sr.dtype).view(1, 3, 1, 1)

    sr_rgb = torch.round((sr + mean).clamp(0.0, 1.0) * 255.0)
    hr_rgb = torch.round((hr + mean).clamp(0.0, 1.0) * 255.0)

    sr_y = rgb_to_y(sr_rgb)
    hr_y = rgb_to_y(hr_rgb)

    sr_y = sr_y[:, :, scale : -scale, scale : -scale]
    hr_y = hr_y[:, :, scale : -scale, scale : -scale]

    psnr = peak_signal_noise_ratio(sr_y, hr_y, 255.0)
    ssim = structural_similarity_index_measure(sr_y, hr_y, True, 1.5, 11, data_range=255.0)

    return psnr, ssim