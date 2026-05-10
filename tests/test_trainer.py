import torch
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import ModelTrainer
from src.models import DenseFilter

def test_trainer_init():
    model = DenseFilter(window_size=10, num_freqs=4)
    trainer = ModelTrainer(model, lr=0.001)
    assert trainer.model == model
    assert len(trainer.history) == 0

def test_trainer_train():
    window_size = 10
    num_freqs = 4
    model = DenseFilter(window_size=window_size, num_freqs=num_freqs)
    trainer = ModelTrainer(model, lr=0.01)
    
    # Create dummy data
    x = torch.randn(10, window_size)
    one_hot = torch.randn(10, num_freqs)
    y = torch.randn(10, window_size)
    
    dataset = TensorDataset(x, one_hot, y)
    dataloader = DataLoader(dataset, batch_size=2)
    
    history = trainer.train(dataloader, epochs=2)
    
    assert len(history) == 2
    assert all(isinstance(loss, float) for loss in history)
    assert len(trainer.history) == 2
