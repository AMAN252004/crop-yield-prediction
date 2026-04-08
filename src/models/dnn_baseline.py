# src/models/dnn_baseline.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_dnn_baseline(
    weather_shape=(52, 6),
    soil_shape=(66,),
    mgmt_shape=(14,),
    dense_units=128,
    dropout_rate=0.2,
    lr=1e-3
):
    """
    Simple DNN baseline model.
    Weather input is flattened (no temporal modeling).
    Soil and management inputs are concatenated with it.
    """

    # WEATHER (flatten)
    input_weather = layers.Input(shape=weather_shape, name="weather_input")
    xw = layers.Flatten()(input_weather)
    xw = layers.Dense(128, activation="relu")(xw)

    # SOIL
    input_soil = layers.Input(shape=soil_shape, name="soil_input")
    xs = layers.Dense(64, activation="relu")(input_soil)

    # MGMT
    input_mgmt = layers.Input(shape=mgmt_shape, name="mgmt_input")
    xm = layers.Dense(32, activation="relu")(input_mgmt)

    # FUSE ALL
    fused = layers.Concatenate()([xw, xs, xm])
    x = layers.Dense(dense_units, activation="relu")(fused)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units // 2, activation="relu")(x)

    out = layers.Dense(1, activation="linear")(x)

    model = models.Model(
        inputs=[input_weather, input_soil, input_mgmt],
        outputs=out
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"]
    )

    print("\n DNN baseline model built.\n")
    model.summary()
    return model