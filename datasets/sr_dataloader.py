import random

import torch
from torch.utils.data import DataLoader

from datasets.sr_dataset import SRDataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)

def create_dataloaders(train_lr_path, train_hr_path,
                       valid_lr_path, valid_hr_path,
                       test_lr_path, test_hr_path,
                       scale, lr_patch_size,
                       batch_size, num_workers, seed, rgb_mean):

    train_dataset = SRDataset(train_lr_path, train_hr_path, "train", scale, lr_patch_size, rgb_mean)
    valid_dataset = SRDataset(valid_lr_path, valid_hr_path, "valid", scale, lr_patch_size, rgb_mean)
    test_dataset = SRDataset(test_lr_path, test_hr_path, "test", scale, lr_patch_size, rgb_mean)

    train_generator = torch.Generator().manual_seed(seed + 1)
    valid_generator = torch.Generator().manual_seed(seed + 2)
    test_generator = torch.Generator().manual_seed(seed + 3)

    train_loader = DataLoader(train_dataset, batch_size, True, generator=train_generator, 
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0, worker_init_fn=seed_worker, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 1, False, generator=valid_generator, 
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, 1, False, generator=test_generator, 
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0, worker_init_fn=seed_worker)

    return train_generator, train_loader, valid_loader, test_loader