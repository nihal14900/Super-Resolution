import random
from pathlib import Path

import numpy as np
import torch

def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, best_psnr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())

    if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state.cpu() for state in checkpoint["cuda_rng"])

    return checkpoint["epoch"], checkpoint["global_step"], checkpoint["best_psnr"]