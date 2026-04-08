# src/models/cnn_gru.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_gru(
    weather_shape=(52, 6),
    soil_shape=(66,),
    mgmt_shape=(14,),
    conv_filters=[32, 64, 128],
    kernel_size=3,
    gru_units=64,
    dense_units=128,
    dropout_rate=0.2,
    lr=1e-3,
):
    """
    CNN + GRU Model for Crop Yield Prediction.
    
    Inputs:
        weather: (52, 6)
        soil: (66,)
        mgmt: (14,)
    """

    # -----------------------------
    # WEATHER BLOCK (CNN)
    # -----------------------------
    input_weather = layers.Input(shape=weather_shape, name="weather_input")
    xw = input_weather

    for f in conv_filters:
        xw = layers.Conv1D(filters=f, kernel_size=kernel_size, activation="relu", padding="same")(xw)
        xw = layers.AveragePooling1D(pool_size=2)(xw)

    xw = layers.Flatten()(xw)
    xw = layers.Dense(64, activation="relu")(xw)
    xw = layers.Dropout(dropout_rate)(xw)

    # -----------------------------
    # SOIL BLOCK (FC)
    # -----------------------------
    input_soil = layers.Input(shape=soil_shape, name="soil_input")
    xs = layers.Dense(128, activation="relu")(input_soil)
    xs = layers.Dropout(dropout_rate)(xs)
    xs = layers.Dense(64, activation="relu")(xs)

    # -----------------------------
    # MANAGEMENT BLOCK (FC)
    # -----------------------------
    input_mgmt = layers.Input(shape=mgmt_shape, name="mgmt_input")
    xm = layers.Dense(32, activation="relu")(input_mgmt)
    xm = layers.Dense(16, activation="relu")(xm)

    # -----------------------------
    # FUSE ALL FEATURES
    # -----------------------------
    fused = layers.Concatenate()([xw, xs, xm])
    fused = layers.Dense(dense_units, activation="relu")(fused)

    # -----------------------------
    # RNN (GRU)
    # -----------------------------
    # Repeat feature vector 5 times (5 years history)
    repeated = layers.RepeatVector(5)(fused)

    gru_out = layers.GRU(gru_units, return_sequences=False)(repeated)

    # -----------------------------
    # FINAL REGRESSION LAYER
    # -----------------------------
    out = layers.Dense(1, activation="linear", name="yield_output")(gru_out)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = models.Model(inputs=[input_weather, input_soil, input_mgmt], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    print("\n CNN + GRU model built successfully.\n")
    model.summary()

    return model