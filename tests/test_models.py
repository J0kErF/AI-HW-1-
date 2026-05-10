import torch

from src.models import DenseFilter, LSTMFilter, RNNFilter


def test_dense_forward():
    model = DenseFilter(10, 4)
    x = torch.randn(32, 10)
    one_hot = torch.randn(32, 4)
    out = model(x, one_hot)
    assert out.shape == (32, 10)

def test_rnn_forward():
    model = RNNFilter(10, 4)
    x = torch.randn(32, 10)
    one_hot = torch.randn(32, 4)
    out = model(x, one_hot)
    assert out.shape == (32, 10)

def test_lstm_forward():
    model = LSTMFilter(10, 4)
    x = torch.randn(32, 10)
    one_hot = torch.randn(32, 4)
    out = model(x, one_hot)
    assert out.shape == (32, 10)
