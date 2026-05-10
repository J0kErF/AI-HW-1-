import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ModelTrainer:
    """Handles training loop for models."""
    
    def __init__(self, model: nn.Module, lr: float):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.history = []

    def train(self, dataloader: DataLoader, epochs: int):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for x, one_hot, y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(x, one_hot)
                loss = self.criterion(predictions, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            self.history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        return self.history