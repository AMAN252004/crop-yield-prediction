# src/models/rnn_baseline.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn_baseline(
    weather_shape=(52, 6),
    soil_shape=(66,),
    mgmt_shape=(14,),
    rnn_units=64,
    dense_units=64,
    dropout_rate=0.2,
    lr=1e-3
):
    """
    Simple RNN baseline.
    Uses SimpleRNN (not LSTM/GRU), purely for comparison.
    """

    # WEATHER → SimpleRNN
    input_weather = layers.Input(shape=weather_shape, name="weather_input")
    xw = layers.SimpleRNN(rnn_units)(input_weather)

    # SOIL
    input_soil = layers.Input(shape=soil_shape, name="soil_input")
    xs = layers.Dense(32, activation="relu")(input_soil)

    # MGMT
    input_mgmt = layers.Input(shape=mgmt_shape, name="mgmt_input")
    xm = layers.Dense(16, activation="relu")(input_mgmt)

    # FUSE
    fused = layers.Concatenate()([xw, xs, xm])
    x = layers.Dense(dense_units, activation="relu")(fused)
    x = layers.Dropout(dropout_rate)(x)

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

    print("\n RNN baseline model built.\n")
    model.summary()
    return model