# src/models/cnn_transformer.py
"""
CNN + Transformer encoder for crop yield prediction.

Inputs:
 - weather: (52, 6)
 - soil: (66,)
 - mgmt: (14,)

Architecture:
 - Shallow 1D CNN to extract local temporal features (keeps reasonable time resolution)
 - Project per-timestep features to embedding_dim
 - Add simple positional encoding (learned)
 - Several Transformer encoder blocks (MultiHeadAttention + FFN)
 - GlobalAveragePooling over time
 - Fuse with soil+mgmt dense features
 - Final dense regression output
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Normalization + MultiHeadAttention + Add
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = layers.Add()([x, inputs])

    # Feed forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    return layers.Add()([x, res])


def build_cnn_transformer(
    weather_shape=(52, 6),
    soil_shape=(66,),
    mgmt_shape=(14,),
    conv_filters=[32, 64],         # shallow CNN
    kernel_size=3,
    embedding_dim=128,             # projection dim for transformer
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=2,
    dropout=0.1,
    dense_units=128,
    lr=3e-4,
):
    # --- Weather CNN (shallow, keep time resolution reasonable) ---
    input_weather = layers.Input(shape=weather_shape, name="weather_input")
    x = input_weather
    for f in conv_filters:
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding="same", activation="relu")(x)
        x = layers.AveragePooling1D(pool_size=2)(x)   # reduces 52 -> 26 -> 13 (reasonable)

    # x shape: (batch, T, channels)
    # Project per-timestep channels to embedding_dim
    x = layers.Dense(embedding_dim, activation="relu")(x)  # (batch, T, embedding_dim)

    # Positional encoding (learned)
    T = x.shape[1]
    if T is None:
        raise ValueError("Time dimension could not be inferred. Check CNN pooling.")
    pos_embedding = layers.Embedding(input_dim=T, output_dim=embedding_dim)
    positions = tf.range(start=0, limit=T, delta=1)
    pos_encoding = pos_embedding(positions)  # (T, embedding_dim)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # (1, T, embedding_dim)
    x = x + pos_encoding  # broadcast over batch

    # --- Transformer encoder blocks ---
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, head_size=embedding_dim // num_heads, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    # Pool across time
    x = layers.GlobalAveragePooling1D()(x)  # (batch, embedding_dim)

    # --- Soil branch ---
    input_soil = layers.Input(shape=soil_shape, name="soil_input")
    xs = layers.Dense(128, activation="relu")(input_soil)
    xs = layers.Dropout(dropout)(xs)
    xs = layers.Dense(64, activation="relu")(xs)

    # --- Management branch ---
    input_mgmt = layers.Input(shape=mgmt_shape, name="mgmt_input")
    xm = layers.Dense(32, activation="relu")(input_mgmt)
    xm = layers.Dense(16, activation="relu")(xm)

    # --- Fuse and final regression ---
    fused = layers.Concatenate()([x, xs, xm])
    fused = layers.Dense(dense_units, activation="relu")(fused)
    fused = layers.Dropout(dropout)(fused)
    fused = layers.Dense(dense_units // 2, activation="relu")(fused)
    out = layers.Dense(1, activation="linear", name="yield_output")(fused)

    model = models.Model(inputs=[input_weather, input_soil, input_mgmt], outputs=out, name="CNN_Transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])

    print("\n Built CNN + Transformer model.")
    model.summary()
    return model