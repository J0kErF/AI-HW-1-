# Signal Filtering with Deep Learning

This project implements and compares three different neural network architectures (Dense, RNN, and LSTM) for the task of extracting pure sine waves from a noisy, multi-frequency combined signal.

## Comparative Analysis of Model Architectures

Recurrent Neural Networks (RNNs) are theoretically well-suited for capturing high-frequency components and rapid signal changes because their hidden state updates at every time step, allowing them to track immediate temporal transitions. However, as the sequence length increases, RNNs suffer significantly from the vanishing gradient problem. During backpropagation through time, the gradients of the loss function tend to decay exponentially as they are multiplied by small weights across many steps. This makes it mathematically difficult for the network to "remember" information from the beginning of a long window, effectively limiting the model's ability to learn long-term dependencies.

Long Short-Term Memory (LSTM) networks were specifically designed to overcome the vanishing gradient limitation through a more complex architecture centered around the Cell State. The Cell State acts as a high-speed conveyor belt that allows information to flow through the sequence with minimal interference. This flow is regulated by three distinct gate mechanisms: the Forget Gate (deciding what to discard), the Input Gate (selecting new information to store), and the Output Gate (filtering the cell state for the next hidden state). By maintaining this stable long-term memory, LSTMs are far superior at modeling low-frequency signals and maintaining context over wide temporal windows where standard RNNs would fail.

The inclusion of random noise in both amplitude and phase significantly impacts the Mean Squared Error (MSE) during training and evaluation. Since the noise added to the amplitude and phase is stochastic and zero-centered, it introduces a variance component to the signal that the model cannot perfectly predict from the input features alone. In terms of the bias-variance tradeoff, this noise creates an "irreducible error" floor. Even an optimal model will have a non-zero MSE because it can only learn the underlying deterministic sine wave patterns, while the random fluctuations remain as residual errors. This forces the model to focus on the robust spectral features of the signal rather than overfitting to transient, noisy artifacts.

## Project Structure
- `src/`: Core logic including data generation, models, and training loops.
- `tests/`: Unit tests for data generation, models, and trainers.
- `results/`: Output plots showing training loss over time.

## Getting Started
To run the training pipeline:
```powershell
$env:PYTHONPATH = ".;$env:PYTHONPATH"; uv run python src/main.py
```

To run tests:
```powershell
$env:PYTHONPATH = ".;$env:PYTHONPATH"; uv run pytest --cov=src
```
