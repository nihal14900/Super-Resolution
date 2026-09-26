import csv
from pathlib import Path

import torch
from torchvision.utils import save_image

from datasets.sr_transform import add_rgb_mean
from datasets.sr_dataset import SRDataset
from datasets.sr_dataloader import DataLoader
from models.edsr import EDSR
from metrics.sr_metrics import calculate_metrics

TEST_LR_PATH = "data/test/set5/lr/bicubic/x2"
TEST_HR_PATH = "data/test/set5/hr"

MODEL_PATH = Path("checkpoints/edsr/best_model.pt")
OUTPUT_DIR = Path("results/edsr")

RGB_MEAN = (0.4488, 0.4371, 0.4040)

NUM_BLOCKS = 16
NUM_FEATURES = 64

SCALE = 2
LR_PATCH_SIZE = 48
BATCH_SIZE = 1

SAVE_IMAGES = True

@torch.inference_mode()
def evaluate(model, dataloader, dataset, rgb_mean, scale, output_dir, save_images):
    model.eval()

    device = next(model.parameters()).device

    output_dir.mkdir(parents=True, exist_ok=True)
    sr_dir = output_dir / "SR"

    if save_images:
        sr_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_psnr = 0.0
    total_ssim = 0.0

    for index, (lr, hr) in enumerate(dataloader):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = model(lr)

        psnr, ssim = calculate_metrics(sr, hr, rgb_mean, scale)

        psnr_value = psnr.item()
        ssim_value = ssim.item()

        image_name = dataset.pairs[index][1].name

        if save_images:
            sr_image = add_rgb_mean(sr, rgb_mean).clamp(0.0, 1.0)
            save_image(sr_image, sr_dir / image_name)

        results.append({
            "image": image_name,
            "psnr": psnr_value,
            "ssim": ssim_value
        })

        total_psnr += psnr_value
        total_ssim += ssim_value

        print(f"{image_name} | PSNR {psnr_value:.4f} dB | SSIM {ssim_value:.6f}")

    mean_psnr = total_psnr / len(results)
    mean_ssim = total_ssim / len(results)

    with (output_dir / "image_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Number of images: {len(results)}")
    print(f"Average PSNR: {mean_psnr}")
    print(f"Average SSIM: {mean_ssim}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = torch.tensor(RGB_MEAN, dtype=torch.float32)

    test_dataset = SRDataset(TEST_LR_PATH, TEST_HR_PATH, "test", SCALE, LR_PATCH_SIZE, rgb_mean)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False, num_workers=0, pin_memory=device.type == "cuda")

    model = EDSR(SCALE, NUM_BLOCKS, NUM_FEATURES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, device, weights_only=True))

    evaluate(model, test_loader, test_dataset, rgb_mean, SCALE, OUTPUT_DIR, SAVE_IMAGES)


if __name__ == "__main__":
    main()

