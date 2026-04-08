# src/train_cnn_transformer_lstm.py
"""
Train script for CNN -> Transformer -> LSTM hybrid.
Run: python -m src.train_cnn_transformer_lstm
"""
import os, json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.dataloader import load_processed, train_val_test_split, fit_scalers, get_tf_datasets
from src.models.cnn_transformer_lstm import build_cnn_transformer_lstm

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train():
    Xw, Xs, Xm, y, meta = load_processed()
    train, val, test = train_val_test_split(Xw, Xs, Xm, y, test_size=0.2, val_size=0.1, random_state=42)
    scalers = fit_scalers(train)
    ds_train, ds_val, ds_test, splits = get_tf_datasets(train, val, test, scalers=scalers, batch_size=32)

    model = build_cnn_transformer_lstm(
        weather_shape=Xw.shape[1:],
        soil_shape=Xs.shape[1:],
        mgmt_shape=Xm.shape[1:],
        conv_filters=[32,64],
        kernel_size=3,
        pool_size=1,               # keep more time resolution
        embedding_dim=128,
        num_heads=8,
        ff_dim=256,
        num_transformer_blocks=3,
        dropout=0.15,
        lstm_units=128,
        dense_units=128,
        lr=3e-4
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
        ModelCheckpoint(os.path.join(RESULTS_DIR, "cnn_transformer_lstm_best.h5"), save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )

    preds = model.predict(ds_test).flatten()
    y_test = splits[2]["y"]

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print("\n============= TEST METRICS (Hybrid CNN-Transformer-LSTM) =============")
    print(metrics)
    with open(os.path.join(RESULTS_DIR, "metrics_cnn_transformer_lstm.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train()

# Save predictions and true values
np.save(f"results/preds_{model.name}.npy", preds)

# Save y_test only once (overwrite is fine)
np.save("results/y_test.npy", y_test)