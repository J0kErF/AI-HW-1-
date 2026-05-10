# Product Requirements Document (PRD)

## 1. Project Overview
Build a Deep Learning system capable of filtering a specific pure sine wave from a noisy, combined signal containing 4 distinct frequencies.

## 2. Goals & KPIs
*   **Goal:** Compare Fully Connected, RNN, and LSTM architectures on a time-series extraction task.
*   **KPI 1:** Test coverage > 85%.
*   **KPI 2:** 0 Ruff linter violations.
*   **KPI 3:** Successful extraction of target frequency with MSE < 0.1.

## 3. Scope & Constraints
*   **Frequencies:** 1Hz, 3Hz, 5Hz, 7Hz.
*   **Noise:** Applied to both amplitude and phase.
*   **Window Size:** 10 samples.
*   **Constraint:** Max 150 lines of code per Python file.