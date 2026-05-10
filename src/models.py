import torch
import torch.nn as nn


class DenseFilter(nn.Module):
    """Fully Connected Network for signal filtering."""
    def __init__(self, window_size: int, num_freqs: int):
        super().__init__()
        input_dim = window_size + num_freqs
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, window_size)
        )

    def forward(self, x, one_hot):
        combined = torch.cat((x, one_hot), dim=1)
        return self.net(combined)

class RNNFilter(nn.Module):
    """RNN Network for signal filtering."""
    def __init__(self, window_size: int, num_freqs: int):
        super().__init__()
        self.window_size = window_size
        # Input per step: 1 signal value + num_freqs one-hot values
        self.rnn = nn.RNN(input_size=1 + num_freqs, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, one_hot):
        # Expand one_hot to match sequence length
        one_hot_seq = one_hot.unsqueeze(1).repeat(1, self.window_size, 1)
        x_seq = x.unsqueeze(-1)

        combined = torch.cat((x_seq, one_hot_seq), dim=2)
        out, _ = self.rnn(combined)
        return self.fc(out).squeeze(-1)

class LSTMFilter(nn.Module):
    """LSTM Network for signal filtering."""
    def __init__(self, window_size: int, num_freqs: int):
        super().__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size=1 + num_freqs, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x, one_hot):
        one_hot_seq = one_hot.unsqueeze(1).repeat(1, self.window_size, 1)
        x_seq = x.unsqueeze(-1)

        combined = torch.cat((x_seq, one_hot_seq), dim=2)
        out, _ = self.lstm(combined)
        return self.fc(out).squeeze(-1)
