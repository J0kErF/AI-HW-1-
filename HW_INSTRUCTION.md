Here is the detailed breakdown of the homework assignment:

### 1. Data Generation (Creating the Dataset)
You must programmatically generate a dataset of sine waves.
*   **The Signals:** Create 4 distinct sine waves with different, constant frequencies (the professor suggested 1Hz, 3Hz, 5Hz, and 7Hz, but you can choose your own as long as they are fixed).
*   **The Formula:** $y = (A \pm \text{noise}) \cdot \sin(2\pi f t + \text{Phase} \pm \text{noise})$
*   **Adding Noise:** You must add random noise to both the Amplitude ($A$) and the Phase of *each individual sine wave* before summing them. The noise should be a percentage of the signal (e.g., $\pm 2\%$, $\pm 5\%$). The phase noise should be between $0$ and $2\pi$.
*   **The Combined Signal:** Sum the 4 noisy sine waves together to create one complex, noisy signal.
*   **Sampling Parameters:** 
    *   Duration: 10 seconds.
    *   Sampling Rate: 1000 Hz (1000 samples per second).
    *   Total size per signal: 10,000 samples.
*   **Context Window:** The network will process the data in chunks. The "Context Window" size is **10 samples**.

### 2. The Core Task
You need to train a model that acts as a smart filter.
*   **Input:** 
    1. A window of 10 samples from the **noisy, combined signal**.
    2. A **One-Hot Encoded vector** (e.g., `[0, 1, 0, 0]`) that acts as a "control button" telling the network *which* of the 4 frequencies it needs to extract.
*   **Target/Output:** A window of 10 samples representing the **pure, noiseless sine wave** of the requested frequency.
*   **Loss Function:** Mean Squared Error (MSE) between the network's prediction and the pure sine wave.

### 3. Neural Network Architectures
You must build, train, and compare **three** different network architectures for this exact same task:
1.  **Fully Connected Network** (Standard Dense Network)
2.  **RNN** (Recurrent Neural Network)
3.  **LSTM** (Long Short-Term Memory)

*(Note: Hyperparameters like the number of layers, perceptrons per layer, and exact noise distributions are left to your "academic freedom" — you must choose them and justify your choices).*

### 4. Analysis and Documentation (The `README` file)
The professor emphasized that the analysis is the most critical part. You must submit a highly detailed `README` file that includes:
*   **Comparative Analysis:** Compare the performance of the Fully Connected, RNN, and LSTM networks. 
*   **Prove the Theory:** Test the professor's claim from the lecture that RNNs are better at capturing high frequencies (fast changes), while LSTMs are better at handling low frequencies (slow changes over time).
*   **Noise vs. Error:** Analyze and show the relationship between the intensity of the noise you added and the resulting error rate (MSE) of the network.
*   **Visuals:** Include screenshots, graphs of the sine waves (pure, noisy, and combined), and error/loss plots.
*   **Justification:** Explain *why* certain architectures succeeded or failed. Convince the professor of your conclusions.

### 5. Technical & Submission Guidelines
*   **Code Constraints:** Python files must not exceed **150 lines of code** per file. You must include **Unit Tests**. Building a GUI is optional.
*   **Self-Grading Mechanism:** You must declare a parameter representing the "strictness" of your grade. If you give yourself a 100, the grader will look for the tiniest mistakes and penalize heavily. If you grade yourself too low (e.g., 60 when you deserve an 80), the system will pull your final grade down. You need to accurately assess your work within $\pm 5$ points.
*   **Submission Format:** A PDF containing your Name, ID, and a link to your code repository (e.g., GitHub), submitted via Moodle.
*   **Groups:** Work in pairs (2 people). Groups of 3 are strictly forbidden (except for military reservists). You can work alone with special permission.
*   **Deadline & Late Policy:** You have two weeks. You can submit late, but it costs **5 points per day of delay** (Reservists are exempt from this and can submit anytime until the final project).