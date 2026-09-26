import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import math
from pathlib import Path

import numpy as np
import torch

from datasets.sr_dataloader import create_dataloaders
from models.edsr import EDSR
from losses.l1_loss import create_loss
from optimizers.adam_optimizer import create_optimizer, create_scheduler
from utils.checkpoint import save_checkpoint, load_checkpoint
from trainer.edsr_trainer import train_step, validate
from utils.metrics_logger import SRMetricLogger

SCALE = 4

TRAIN_LR_PATH = f"data/train/div2k/lr/bicubic/x{SCALE}"
TRAIN_HR_PATH = "data/train/div2k/hr"
VALID_LR_PATH = f"data/valid/div2k/lr/bicubic/x{SCALE}"
VALID_HR_PATH = "data/valid/div2k/hr"
TEST_LR_PATH = f"data/test/set5/lr/bicubic/x{SCALE}"
TEST_HR_PATH = "data/test/set5/hr"

RGB_MEAN = (0.4488, 0.4371, 0.4040)

NUM_BLOCKS = 16
NUM_FEATURES = 64

LR_PATCH_SIZE = 64
BATCH_SIZE = 16

LEARNING_RATE = 1e-4
MAX_UPDATES = 100_000

STEP_SIZE = 100_000
GAMMA = 0.5

VALIDATE_EVERY = 1_000
SAVE_EVERY_EPOCHS = 1
LOG_EVERY = 1_000

SEED = 42
NUM_WORKERS = 0

OUTPUT_DIR = Path("checkpoints/edsr")
LOG_DIR = "experiments/edsr"
RESUME_PATH = "checkpoints/edsr/checkpoint_34000.pt" 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_gpu_memory(stage):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    print(f"{stage} | Allocated: {allocated:2f} GB | Reserved: {reserved:.2f} GB")

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = torch.tensor(RGB_MEAN, dtype=torch.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_generator, train_loader, valid_loader, _ = create_dataloaders(TRAIN_LR_PATH, TRAIN_HR_PATH,
                                                                             VALID_LR_PATH, VALID_HR_PATH,
                                                                             TEST_LR_PATH, TEST_HR_PATH,
                                                                             SCALE, LR_PATCH_SIZE, BATCH_SIZE, NUM_WORKERS, SEED, rgb_mean)

    model = EDSR(SCALE, NUM_BLOCKS, NUM_FEATURES).to(device)
    criterion = create_loss()
    optimizer = create_optimizer(model, LEARNING_RATE)
    scheduler = create_scheduler(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    start_epoch = 0
    global_step = 0
    best_psnr = -math.inf

    if RESUME_PATH is not None:
        start_epoch, global_step, best_psnr = load_checkpoint(RESUME_PATH, model, optimizer, scheduler, device)

    logger = SRMetricLogger(LOG_DIR, global_step)

    epoch = start_epoch

    while global_step < MAX_UPDATES:
        epoch += 1

        train_generator.manual_seed(SEED + epoch)

        for lr, hr in train_loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            train_loss = train_step(model, lr, hr, criterion, optimizer, scheduler)
            global_step += 1

            if global_step == 1 or global_step % LOG_EVERY == 0:
                logger.log_train(global_step, train_loss, optimizer.param_groups[0]["lr"])
                print(f"Epoch {epoch} | Update {global_step}/{MAX_UPDATES} | L1 {train_loss:.6f} | LR {optimizer.param_groups[0]["lr"]:.8f}")

            if global_step % VALIDATE_EVERY == 0 or global_step == MAX_UPDATES:

                # print_gpu_memory("Before validation")

                valid_loss, valid_psnr, valid_ssim = validate(model, valid_loader, criterion, device, rgb_mean, SCALE)

                # print_gpu_memory("After validation")

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                    # print_gpu_memory("After cleaning cache")

                logger.log_valid(global_step, valid_loss, valid_psnr, valid_ssim)

                print(f"Validation | Update {global_step} | L1 {valid_loss:.6f} | PSNR {valid_psnr:.4f} dB | SSIM {valid_ssim:.6f}")
                
                if valid_psnr > best_psnr:
                    best_psnr = valid_psnr
                    filename2 = f"best_{global_step}.pt"
                    torch.save(model.state_dict(), OUTPUT_DIR / filename2)

                filename1 = f"checkpoint_{global_step}.pt"
                save_checkpoint(OUTPUT_DIR / filename1, model, optimizer, scheduler, epoch, global_step, best_psnr)

            if global_step >= MAX_UPDATES:
                break

        # if epoch % SAVE_EVERY_EPOCHS == 0 or global_step >= MAX_UPDATES:
        #     save_checkpoint(OUTPUT_DIR / "latest_checkpoint.pt", model, optimizer, scheduler, epoch, global_step, best_psnr)

    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")

    print("Training completed.")

    logger.close()

if __name__ == "__main__":
    main()
