# src/models/cnn_lstm_attention.py

import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------------------------------
# Correct Attention Layer
# ------------------------------------------------------
class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        self.W = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True
        )
        self.v = self.add_weight(
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        score = tf.tensordot(score, self.v, axes=1)  # (batch, time, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# ------------------------------------------------------
# Main CNN + LSTM + Attention Model
# ------------------------------------------------------
def build_cnn_lstm_attention(
    weather_shape=(52, 6),
    soil_shape=(66,),
    mgmt_shape=(14,),
    conv_filters=[32, 64],
    kernel_size=3,
    lstm_units=128,
    dense_units=128,
    dropout_rate=0.25,
    lr=3e-4
):
    # WEATHER INPUT
    input_weather = layers.Input(shape=weather_shape, name="weather_input")
    xw = input_weather

    # CNN (mild pooling)
    for f in conv_filters:
        xw = layers.Conv1D(f, kernel_size, padding='same', activation='relu')(xw)
        xw = layers.AveragePooling1D(pool_size=2)(xw)

    xw = layers.Dropout(dropout_rate)(xw)

    # LSTM
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(xw)

    # ATTENTION
    attn_out = SimpleAttention()(lstm_out)

    # SOIL
    input_soil = layers.Input(shape=soil_shape, name="soil_input")
    xs = layers.Dense(128, activation='relu')(input_soil)
    xs = layers.Dropout(dropout_rate)(xs)
    xs = layers.Dense(64, activation='relu')(xs)

    # MGMT
    input_mgmt = layers.Input(shape=mgmt_shape, name="mgmt_input")
    xm = layers.Dense(32, activation='relu')(input_mgmt)
    xm = layers.Dense(16, activation='relu')(xm)

    # FUSE
    fused = layers.Concatenate()([attn_out, xs, xm])
    x = layers.Dense(dense_units, activation='relu')(fused)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units // 2, activation='relu')(x)

    out = layers.Dense(1, activation='linear')(x)

    model = models.Model(
        inputs=[input_weather, input_soil, input_mgmt],
        outputs=out
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='mse',
        metrics=['mae']
    )

    print("\n Corrected CNN + LSTM + Attention model built.\n")
    model.summary()
    return model