from src.config import Config
from src.data_generator import SignalDataset


def test_dataset_generation():
    cfg = Config([1, 3], 100, 1, 10, 0.05, 1, 0.01, 32)
    dataset = SignalDataset(cfg)

    assert len(dataset.pure_signals) == 2
    assert len(dataset.combined_signal) == 100

    x, one_hot, y = dataset[0]
    assert x.shape == (10,)
    assert one_hot.shape == (2,)
    assert y.shape == (10,)
