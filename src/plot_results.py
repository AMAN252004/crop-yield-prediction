# src/plot_results.py

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"

# -------------------------------------------------------
# LOAD ALL RESULTS
# -------------------------------------------------------
models = [
    ("DNN", "metrics_dnn.json"),
    ("Simple RNN", "metrics_rnn.json"),
    ("CNN-GRU", "metrics_cnn_gru.json"),
    ("CNN-BiLSTM", "metrics_cnn_bilstm.json"),
    ("CNN-LSTM-Attn", "metrics_cnn_lstm_attn.json"),
    ("CNN-Transformer", "metrics_cnn_transformer.json"),
    ("Hybrid CNN-Transformer-LSTM", "metrics_cnn_transformer_lstm.json")
]

data = []
for model_name, file in models:
    path = os.path.join(RESULTS_DIR, file)
    if not os.path.exists(path):
        print(f"⚠ Warning: {file} not found, skipping.")
        continue
    with open(path, "r") as f:
        metrics = json.load(f)
        data.append([model_name, metrics["rmse"], metrics["mae"], metrics["r2"]])

df = pd.DataFrame(data, columns=["Model", "RMSE", "MAE", "R2"])
print("\n==================== MODEL COMPARISON ====================")
print(df)
print("==========================================================\n")

os.makedirs("results/plots", exist_ok=True)

# -------------------------------------------------------
# BAR PLOTS (RMSE, MAE, R²)
# -------------------------------------------------------
plt.figure(figsize=(12,6))
plt.bar(df["Model"], df["RMSE"], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("RMSE")
plt.title("Model Comparison (RMSE)")
plt.tight_layout()
plt.savefig("results/plots/rmse_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(12,6))
plt.bar(df["Model"], df["MAE"], color='lightgreen')
plt.xticks(rotation=45, ha='right')
plt.ylabel("MAE")
plt.title("Model Comparison (MAE)")
plt.tight_layout()
plt.savefig("results/plots/mae_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(12,6))
plt.bar(df["Model"], df["R2"], color='salmon')
plt.xticks(rotation=45, ha='right')
plt.ylabel("R² Score")
plt.title("Model Comparison (R²)")
plt.tight_layout()
plt.savefig("results/plots/r2_comparison.png", dpi=300)
plt.close()

# -------------------------------------------------------
# LINE COMPARISON
# -------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(df["Model"], df["RMSE"], marker='o', label="RMSE")
plt.plot(df["Model"], df["MAE"], marker='o', label="MAE")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Score Value")
plt.title("RMSE & MAE Line Comparison Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/line_rmse_mae.png", dpi=300)
plt.close()

# -------------------------------------------------------
# RANKING TABLE (EXPORT AS CSV)
# -------------------------------------------------------
df_sorted = df.sort_values("RMSE")
df_sorted.to_csv("results/plots/model_ranking.csv", index=False)

# -------------------------------------------------------
# SCATTER & RESIDUALS REQUIRES ACTUAL VS PREDICTED ARRAYS
# -------------------------------------------------------
# If needed we can load predictions saved by training scripts
# For now, generate placeholder example code for your best model

print("All comparison plots generated and saved in: results/plots/")
np.save(f"results/preds_{model_name}.npy", preds)
np.save(f"results/ytest.npy", y_test)