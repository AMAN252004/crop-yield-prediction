# src/train_cnn_bilstm.py

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dataloader import load_processed, train_val_test_split, fit_scalers, get_tf_datasets
from src.models.cnn_bilstm import build_cnn_bilstm


def train():
    Xw, Xs, Xm, y, meta = load_processed()

    train, val, test = train_val_test_split(Xw, Xs, Xm, y)

    scalers = fit_scalers(train)

    ds_train, ds_val, ds_test, splits = get_tf_datasets(
        train, val, test, scalers=scalers, batch_size=32
    )

    model = build_cnn_bilstm(
        weather_shape=Xw.shape[1:],
        soil_shape=Xs.shape[1:],
        mgmt_shape=Xm.shape[1:],
        lstm_units=64,
    )

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5),
        ModelCheckpoint("results/cnn_bilstm_best.h5",
                        save_best_only=True, monitor="val_loss"),
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    preds = model.predict(ds_test).flatten()
    y_test = splits[2]["y"]

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n============= TEST METRICS (BiLSTM) =============")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R²  :", r2)
    print("==================================================")

if __name__ == "__main__":
    train()

# Save predictions and true values
np.save(f"results/preds_{model.name}.npy", preds)

# Save test labels ONCE (overwriting is fine)
np.save("results/y_test.npy", y_test)