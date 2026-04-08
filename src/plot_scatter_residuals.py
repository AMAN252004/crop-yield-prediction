# src/plot_scatter_residuals.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Models to load predictions for
MODEL_NAMES = [
    "DNN",
    "SimpleRNN",
    "CNN_GRU",
    "CNN_BiLSTM",
    "CNN_LSTM_Attention",
    "CNN_Transformer",
    "CNN_Transformer_LSTM"
]

# ----------------------------------------------
# Load true y_test values
# ----------------------------------------------
y_test_path = os.path.join(RESULTS_DIR, "y_test.npy")
if not os.path.exists(y_test_path):
    raise FileNotFoundError("y_test.npy missing! Did you add saving inside train scripts?")

y_test = np.load(y_test_path)

# ----------------------------------------------
# Generate Scatter and Residual plots
# ----------------------------------------------

for model in MODEL_NAMES:
    pred_path = os.path.join(RESULTS_DIR, f"preds_{model}.npy")

    if not os.path.exists(pred_path):
        print(f"⚠ Skipping {model}, no prediction file found.")
        continue

    preds = np.load(pred_path)

    # Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, preds, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"Actual vs Predicted — {model}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scatter_{model}.png", dpi=300)
    plt.close()

    # Residual Plot
    residuals = y_test - preds
    plt.figure(figsize=(7,7))
    plt.scatter(preds, residuals, alpha=0.5, s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Yield")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residuals Plot — {model}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/residuals_{model}.png", dpi=300)
    plt.close()

    print(f"Generated scatter & residuals plots for {model}")

print("\nAll plots saved under: results/plots/")