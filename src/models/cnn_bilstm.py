# src/models/cnn_bilstm.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_bilstm(
    weather_shape=(52, 6),
    soil_shape=(66,),
    mgmt_shape=(14,),
    conv_filters=[32, 64, 128],
    kernel_size=3,
    lstm_units=64,
    dense_units=128,
    dropout_rate=0.2,
    lr=1e-3,
):
    """
    CNN + BiLSTM architecture for Crop Yield Prediction.
    """

    # -----------------------------
    # WEATHER BLOCK (CNN)
    # -----------------------------
    input_weather = layers.Input(shape=weather_shape, name="weather_input")
    xw = input_weather

    for f in conv_filters:
        xw = layers.Conv1D(filters=f, kernel_size=kernel_size,
                           activation="relu", padding="same")(xw)
        xw = layers.AveragePooling1D(pool_size=2)(xw)

    xw = layers.Flatten()(xw)
    xw = layers.Dense(64, activation="relu")(xw)
    xw = layers.Dropout(dropout_rate)(xw)

    # -----------------------------
    # SOIL BLOCK
    # -----------------------------
    input_soil = layers.Input(shape=soil_shape, name="soil_input")
    xs = layers.Dense(128, activation="relu")(input_soil)
    xs = layers.Dropout(dropout_rate)(xs)
    xs = layers.Dense(64, activation="relu")(xs)

    # -----------------------------
    # MGMT BLOCK
    # -----------------------------
    input_mgmt = layers.Input(shape=mgmt_shape, name="mgmt_input")
    xm = layers.Dense(32, activation="relu")(input_mgmt)
    xm = layers.Dense(16, activation="relu")(xm)

    # -----------------------------
    # FUSION
    # -----------------------------
    fused = layers.Concatenate()([xw, xs, xm])
    fused = layers.Dense(dense_units, activation="relu")(fused)

    # -----------------------------
    # BiLSTM
    # -----------------------------
    repeated = layers.RepeatVector(5)(fused)  # 5-year dependency
    bilstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False)
    )(repeated)

    # -----------------------------
    # FINAL REGRESSION
    # -----------------------------
    out = layers.Dense(1, activation="linear", name="yield_output")(bilstm_out)

    model = models.Model(
        inputs=[input_weather, input_soil, input_mgmt],
        outputs=out
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    print("\n CNN + BiLSTM model built successfully.\n")
    model.summary()

    return model