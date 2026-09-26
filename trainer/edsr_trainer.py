import math

import torch

from metrics.sr_metrics import calculate_metrics

def train_step(model, lr, hr, criterion, optimizer, scheduler):
    model.train()
    optimizer.zero_grad(True)

    sr = model(lr)

    loss = criterion(sr, hr)
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()

@torch.no_grad()
def validate(model, dataloder, criterion, device, rgb_mean, scale):
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_elements = 0
    total_images = 0

    for lr, hr in dataloder:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = model(lr)

        total_loss += criterion(sr, hr).item() * hr.numel()
        total_elements += hr.numel()

        for prediction, target in zip(sr, hr):
            psnr, ssim = calculate_metrics(prediction.unsqueeze(0), target.unsqueeze(0), rgb_mean, scale)

            total_psnr += psnr.item()
            total_ssim += ssim.item()
            total_images += 1

    return total_loss / total_elements, total_psnr / total_images, total_ssim / total_images
