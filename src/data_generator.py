import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """Generates noisy combined signals and pure target signals."""

    def __init__(self, config):
        self.cfg = config
        self.t = np.linspace(0, self.cfg.duration, self.cfg.duration * self.cfg.sample_rate)
        self.num_samples = len(self.t)
        self.pure_signals, self.combined_signal = self._generate_signals()
        self.windows, self.targets, self.one_hots = self._create_windows()

    def _generate_signals(self):
        pure_signals = []
        combined_signal = np.zeros(self.num_samples)

        for f in self.cfg.frequencies:
            # Pure signal
            pure = np.sin(2 * np.pi * f * self.t)
            pure_signals.append(pure)

            # Noisy signal: y = (A +- noise) * sin(2*pi*f*t + Phase +- noise)
            amp_noise = np.random.uniform(-self.cfg.noise_level, self.cfg.noise_level, self.num_samples)
            phase_noise = np.random.uniform(0, 2 * np.pi, self.num_samples) * self.cfg.noise_level

            noisy = (1.0 + amp_noise) * np.sin(2 * np.pi * f * self.t + phase_noise)
            combined_signal += noisy

        return pure_signals, combined_signal

    def _create_windows(self):
        windows, targets, one_hots = [], [], []
        num_freqs = len(self.cfg.frequencies)

        # Slide window
        for i in range(self.num_samples - self.cfg.window_size):
            window = self.combined_signal[i : i + self.cfg.window_size]

            # For each window, create a sample for EACH target frequency
            for freq_idx in range(num_freqs):
                target = self.pure_signals[freq_idx][i : i + self.cfg.window_size]

                one_hot = np.zeros(num_freqs)
                one_hot[freq_idx] = 1.0

                windows.append(window)
                targets.append(target)
                one_hots.append(one_hot)

        return np.array(windows), np.array(targets), np.array(one_hots)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        one_hot = torch.tensor(self.one_hots[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, one_hot, y
