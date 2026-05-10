"""
Joint classification + denoising of noisy sine waves.

Setup:
  - 4 classes: sine waves at 50, 100, 200, 400 Hz
  - Sample rate: 1000 Hz
  - Signal length: 512 samples (~0.512 s)
  - Noise: additive Gaussian white noise, SNR sampled per-example in [0, 15] dB
  - Model: 1D U-Net with classification head off the bottleneck
  - Loss: MSE(denoised, clean) + lambda * CE(logits, class)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
SR = 1000               # Hz
N = 512                 # samples per signal
FREQS = [50, 100, 200, 400]
NUM_CLASSES = len(FREQS)
SNR_DB_RANGE = (0.0, 15.0)
BATCH = 64
EPOCHS = 20
LR = 1e-3
LAMBDA_CLS = 0.5        # weight for classification loss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Data
# -----------------------------
class SineDataset(Dataset):
    """Generates (noisy, clean, class_idx) on the fly."""

    def __init__(self, n_samples: int, sr: int = SR, length: int = N,
                 freqs=FREQS, snr_db_range=SNR_DB_RANGE):
        self.n_samples = n_samples
        self.sr = sr
        self.length = length
        self.freqs = freqs
        self.snr_db_range = snr_db_range
        self.t = np.arange(length) / sr  # time vector

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Pick class
        cls = np.random.randint(len(self.freqs))
        f = self.freqs[cls]

        # Random phase so the model can't cheat on alignment
        phase = np.random.uniform(0, 2 * math.pi)
        clean = np.sin(2 * math.pi * f * self.t + phase).astype(np.float32)
        # clean has unit amplitude => signal power = 0.5

        # Sample SNR (dB), compute noise std
        snr_db = np.random.uniform(*self.snr_db_range)
        sig_power = 0.5
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.randn(self.length).astype(np.float32) * math.sqrt(noise_power)

        noisy = clean + noise

        # Shape: (1, length) for Conv1d
        return (
            torch.from_numpy(noisy).unsqueeze(0),
            torch.from_numpy(clean).unsqueeze(0),
            torch.tensor(cls, dtype=torch.long),
        )


# -----------------------------
# Model: 1D U-Net + classifier head
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, base=32, num_classes=NUM_CLASSES):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, base)            # 512
        self.enc2 = ConvBlock(base, base * 2)     # 256
        self.enc3 = ConvBlock(base * 2, base * 4) # 128
        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base * 4, base * 8)  # 64

        # Decoder
        self.up3 = nn.ConvTranspose1d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose1d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_denoise = nn.Conv1d(base, 1, kernel_size=1)

        # Classification head from bottleneck features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        e1 = self.enc1(x)                  # (B, base, 512)
        e2 = self.enc2(self.pool(e1))      # (B, 2B, 256)
        e3 = self.enc3(self.pool(e2))      # (B, 4B, 128)
        b  = self.bottleneck(self.pool(e3))# (B, 8B, 64)

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        denoised = self.out_denoise(d1)    # (B, 1, 512)
        logits = self.classifier(b)        # (B, num_classes)
        return denoised, logits


# -----------------------------
# Train / eval
# -----------------------------
def run():
    train_ds = SineDataset(n_samples=8000)
    val_ds   = SineDataset(n_samples=1000)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    model = UNet1D().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tr_loss = tr_mse = tr_ce = tr_correct = tr_total = 0
        for noisy, clean, cls in train_dl:
            noisy, clean, cls = noisy.to(DEVICE), clean.to(DEVICE), cls.to(DEVICE)
            denoised, logits = model(noisy)
            loss_mse = mse(denoised, clean)
            loss_ce  = ce(logits, cls)
            loss = loss_mse + LAMBDA_CLS * loss_ce

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = cls.size(0)
            tr_loss += loss.item() * bs
            tr_mse  += loss_mse.item() * bs
            tr_ce   += loss_ce.item() * bs
            tr_correct += (logits.argmax(1) == cls).sum().item()
            tr_total += bs

        # ---- val ----
        model.eval()
        v_mse = v_correct = v_total = 0
        with torch.no_grad():
            for noisy, clean, cls in val_dl:
                noisy, clean, cls = noisy.to(DEVICE), clean.to(DEVICE), cls.to(DEVICE)
                denoised, logits = model(noisy)
                v_mse += mse(denoised, clean).item() * cls.size(0)
                v_correct += (logits.argmax(1) == cls).sum().item()
                v_total += cls.size(0)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss/tr_total:.4f} "
            f"(mse {tr_mse/tr_total:.4f}, ce {tr_ce/tr_total:.4f}, "
            f"acc {tr_correct/tr_total:.3f}) | "
            f"val mse {v_mse/v_total:.4f}, acc {v_correct/v_total:.3f}"
        )

    return model


if __name__ == "__main__":
    run()
