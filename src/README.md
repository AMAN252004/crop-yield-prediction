crop_yield_new/
│
├── data/
│   ├── processed/
│   │   ├── Xw.npy               # Weather tensor (N, 52, 6)
│   │   ├── Xs.npy               # Soil features (N, 66)
│   │   ├── Xm.npy               # Management features (N, 14)
│   │   ├── y.npy                # Yield targets (N,)
│   │   ├── meta.json            # Detected feature metadata
│   │
│   ├── splits/
│   │   ├── idx_train.npy
│   │   ├── idx_val.npy
│   │   ├── idx_test.npy
│
├── results/
│   ├── preds_*.npy              # Saved predictions for all models
│   ├── y_test.npy               # True test labels
│   ├── metrics_*.json           # RMSE/MAE/R² for each model
│   ├── plots/
│       ├── rmse_comparison.png
│       ├── mae_comparison.png
│       ├── r2_comparison.png
│       ├── line_rmse_mae.png
│       ├── scatter_*.png
│       ├── residuals_*.png
│       ├── model_ranking.csv
│
├── src/
│   ├── dataloader.py            # Loads arrays, splits, scaling, TF datasets
│   ├── inspect_data.py          # Inspects raw CSV structure (initial stage)
│   │
│   ├── models/
│   │   ├── dnn_baseline.py
│   │   ├── rnn_baseline.py
│   │   ├── cnn_gru.py
│   │   ├── cnn_bilstm.py
│   │   ├── cnn_lstm_attention.py
│   │   ├── cnn_transformer.py
│   │   ├── cnn_transformer_lstm.py   # Hybrid model (best performance)
│   │
│   ├── train_dnn.py
│   ├── train_rnn.py
│   ├── train_cnn_gru.py
│   ├── train_cnn_bilstm.py
│   ├── train_cnn_lstm_attention.py
│   ├── train_cnn_transformer.py
│   ├── train_cnn_transformer_lstm.py
│   │
│   ├── plot_results.py           # Bar charts, line plots, rankings
│   ├── plot_scatter_residuals.py # All scatter + residual plots
│   ├── create_y_test.py          # Helper script (saves y_test from dataloader)
│
├── venv/                         # Python virtual environment
│
├── README.md                     # (You are creating this file right now)
│
└── requirements.txt              # (optional if you choose to add)