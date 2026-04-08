# src/train_dnn.py

import os, json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.dataloader import load_processed, train_val_test_split, fit_scalers, get_tf_datasets
from src.models.dnn_baseline import build_dnn_baseline

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train():
    Xw, Xs, Xm, y, meta = load_processed()
    train, val, test = train_val_test_split(Xw, Xs, Xm, y)

    scalers = fit_scalers(train)
    ds_train, ds_val, ds_test, splits = get_tf_datasets(train, val, test, scalers=scalers)

    model = build_dnn_baseline(
        weather_shape=Xw.shape[1:],
        soil_shape=Xs.shape[1:],
        mgmt_shape=Xm.shape[1:]
    )

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint("results/dnn_baseline_best.h5", save_best_only=True)
    ]

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=40,
        callbacks=callbacks,
        verbose=1
    )

    preds = model.predict(ds_test).flatten()
    y_test = splits[2]['y']

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))

    print("\n============= DNN BASELINE RESULTS =============")
    print({"rmse": rmse, "mae": mae, "r2": r2})
    print("=================================================")

    with open("results/metrics_dnn.json", "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f, indent=2)

if __name__ == "__main__":
    train()

# Save predictions and true values
np.save(f"results/preds_{model.name}.npy", preds)

# Save y_test only once (overwrite is fine)
np.save("results/y_test.npy", y_test)