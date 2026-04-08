# src/train_cnn_gru.py

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataloader import load_processed, train_val_test_split, fit_scalers, get_tf_datasets
from models.cnn_gru import build_cnn_gru


def train():
    # Load arrays
    Xw, Xs, Xm, y, meta = load_processed()

    # Split into train/val/test
    train, val, test = train_val_test_split(Xw, Xs, Xm, y)

    # Fit scalers
    scalers = fit_scalers(train)

    # Create tf.data datasets
    ds_train, ds_val, ds_test, splits = get_tf_datasets(train, val, test, scalers=scalers, batch_size=32)

    # Build model
    model = build_cnn_gru(
        weather_shape=Xw.shape[1:],
        soil_shape=Xs.shape[1:],
        mgmt_shape=Xm.shape[1:],
        gru_units=64
    )

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=5, factor=0.5),
        ModelCheckpoint("results/cnn_gru_best.h5", save_best_only=True, monitor="val_loss")
    ]

    # Train
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    preds = model.predict(ds_test).flatten()
    y_test = splits[2]["y"]

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    print("\n================ TEST METRICS ================")
    print("RMSE: ", rmse)
    print("MAE : ", mae)
    print("R²  : ", r2)
    print("==============================================")

if __name__ == "__main__":
    train()

# Save predictions and true values
np.save(f"results/preds_{model.name}.npy", preds)

# Save test labels ONCE (overwriting is fine)
np.save("results/y_test.npy", y_test)