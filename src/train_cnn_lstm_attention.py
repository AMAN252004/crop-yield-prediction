# src/train_cnn_lstm_attention.py
"""
Train script for the CNN + LSTM + Attention model.
Run with: python -m src.train_cnn_lstm_attention
"""

import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.dataloader import load_processed, train_val_test_split, fit_scalers, get_tf_datasets
from src.models.cnn_lstm_attention import build_cnn_lstm_attention

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train():
    Xw, Xs, Xm, y, meta = load_processed()

    # split (will save indices)
    train, val, test = train_val_test_split(Xw, Xs, Xm, y, test_size=0.2, val_size=0.1, random_state=42)

    scalers = fit_scalers(train)
    ds_train, ds_val, ds_test, splits = get_tf_datasets(train, val, test, scalers=scalers, batch_size=32)

    # Build model tuned to your data shapes
    model = build_cnn_lstm_attention(
        weather_shape=Xw.shape[1:],
        soil_shape=Xs.shape[1:],
        mgmt_shape=Xm.shape[1:],
        lstm_units=128,
        dense_units=128,
        dropout_rate=0.25,
        lr=5e-4
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ModelCheckpoint(os.path.join(RESULTS_DIR, "cnn_lstm_attn_best.h5"), save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=80,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test
    preds = model.predict(ds_test).flatten()
    y_test = splits[2]['y']
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n============= TEST METRICS (CNN + LSTM + Attention) =============")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R²  :", r2)
    print("===============================================================")

    # Save metrics to file
    import json
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    with open(os.path.join(RESULTS_DIR, "metrics_cnn_lstm_attn.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train()

# Save predictions and true values
np.save(f"results/preds_{model.name}.npy", preds)

# Save y_test only once (overwrite is fine)
np.save("results/y_test.npy", y_test)