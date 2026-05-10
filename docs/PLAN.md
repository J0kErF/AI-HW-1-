# Architecture & Planning

## 1. System Architecture
*   **Data Layer:** Generates 10 seconds of data at 1000Hz. Applies noise to amplitude and phase.
*   **Model Layer:** PyTorch implementations of Dense, RNN, and LSTM.
*   **Training Layer:** MSE Loss, Adam Optimizer, batch processing.

## 2. Data Flow
1. Generate 4 noisy sine waves.
2. Sum them into a combined signal.
3. Extract 10-sample windows.
4. Append One-Hot encoded target vector.
5. Predict 10-sample pure sine wave.TODO.md