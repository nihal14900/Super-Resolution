import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

class SRMetricLogger:
    def __init__(self, log_dir, resume_step):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir / "tensorboard", purge_step = resume_step + 1)

        self.train_file = log_dir / "train_metrics.csv"
        self.valid_file = log_dir / "valid_metrics.csv"

        train_fields = ["step", "loss", "lr"]
        valid_fields = ["step", "loss", "psnr", "ssim"]

        self._prepare(self.train_file, train_fields, resume_step)
        self._prepare(self.valid_file, valid_fields, resume_step)

    def _prepare(self, path, fields, resume_step):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)
            return

        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))

        rows = [row for row in rows if int(row["step"]) <= resume_step]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)

    def _append(self, path, values):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(values)

    def log_train(self, step, loss, lr):
        self._append(self.train_file, [step, loss, lr])

        self.writer.add_scalar("Train/Loss", loss, step)
        self.writer.add_scalar("Train/Learning_Rate", lr, step)

    def log_valid(self, step, loss, psnr, ssim):
        self._append(self.valid_file, [step, loss, psnr, ssim])

        self.writer.add_scalar("Valid/Loss", loss, step)
        self.writer.add_scalar("Valid/PSNR", psnr, step)
        self.writer.add_scalar("Valid/SSIM", ssim, step)

        self.writer.flush()

    def close(self):
        self.writer.close()