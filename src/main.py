import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.config import load_config
from src.data_generator import SignalDataset
from src.models import DenseFilter, RNNFilter, LSTMFilter
from src.trainer import ModelTrainer

def plot_results(history, name):
    plt.figure()
    plt.plot(history, label="Training Loss")
    plt.title(f"{name} Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(f"results/{name}_loss.png")
    plt.close()

def main():
    print("Loading config and generating data...")
    cfg = load_config()
    # Reduce duration for faster training during development
    cfg.duration = 1 
    dataset = SignalDataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    num_freqs = len(cfg.frequencies)
    
    models = {
        "Dense": DenseFilter(cfg.window_size, num_freqs),
        "RNN": RNNFilter(cfg.window_size, num_freqs),
        "LSTM": LSTMFilter(cfg.window_size, num_freqs)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name} Model...")
        trainer = ModelTrainer(model, cfg.learning_rate)
        history = trainer.train(loader, cfg.epochs)
        plot_results(history, name)
        print(f"{name} training complete. Graph saved to results/.")

if __name__ == "__main__":
    main()