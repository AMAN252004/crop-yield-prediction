# src/models/cnn_transformer_lstm.py
"""
Hybrid CNN -> Transformer -> LSTM model for crop yield prediction.

Inputs:
 - weather: (52, 6)
 - soil:    (66,)
 - mgmt:    (14,)

Output:
 - scalar yield prediction
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    # x: (batch, T, dim)
    # multi-head attention block with residuals and FFN
    attn = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(attn, attn)
    attn = layers.Dropout(dropout)(attn)
    x = layers.Add()([x, attn])

    ffn = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = layers.Dense(ff_dim, activation="relu")(ffn)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    x = layers.Add()([x, ffn])
    return x

def build_cnn_transformer_lstm(
    weather_shape=(52,6),
    soil_shape=(66,),
    mgmt_shape=(14,),
    # CNN params
    conv_filters=[32,64],
    kernel_size=3,
    pool_size=1,               # keep more time steps; set 1 or 2
    # Transformer params
    embedding_dim=128,
    num_heads=8,
    ff_dim=256,
    num_transformer_blocks=2,
    dropout=0.15,
    # LSTM params
    lstm_units=128,
    # Dense
    dense_units=128,
    lr=3e-4
):
    # ---------------- Weather CNN ----------------
    input_weather = layers.Input(shape=weather_shape, name="weather_input")
    x = input_weather
    # shallow CNN but avoid overly compressing time
    for f in conv_filters:
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding="same", activation="relu")(x)
        if pool_size > 1:
            x = layers.AveragePooling1D(pool_size=pool_size)(x)
    # project to embedding_dim
    x = layers.Dense(embedding_dim, activation="relu")(x)  # (batch, T, embedding_dim)

    # positional embedding (learned)
    T = x.shape[1]
    if T is None:
        raise ValueError("Time dimension must be statically known. Check CNN pooling.")
    pos_emb = layers.Embedding(input_dim=T, output_dim=embedding_dim)
    positions = tf.range(start=0, limit=T, delta=1)
    pos_encoding = pos_emb(positions)  # (T, embed)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # (1, T, embed)
    x = x + pos_encoding  # broadcast

    # ---------------- Transformer Encoder Blocks ----------------
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, head_size=embedding_dim//num_heads, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    # Optionally keep sequence for LSTM
    # x shape: (batch, T', embedding_dim)
    # ---------------- LSTM on transformer outputs ----------------
    lstm_out = layers.LSTM(lstm_units, return_sequences=False)(x)  # (batch, lstm_units)

    # ---------------- Soil & Mgmt branches ----------------
    input_soil = layers.Input(shape=soil_shape, name="soil_input")
    xs = layers.Dense(128, activation="relu")(input_soil)
    xs = layers.Dropout(dropout)(xs)
    xs = layers.Dense(64, activation="relu")(xs)

    input_mgmt = layers.Input(shape=mgmt_shape, name="mgmt_input")
    xm = layers.Dense(32, activation="relu")(input_mgmt)
    xm = layers.Dense(16, activation="relu")(xm)

    # ---------------- Fuse ----------------
    fused = layers.Concatenate()([lstm_out, xs, xm])
    x_final = layers.Dense(dense_units, activation="relu")(fused)
    x_final = layers.Dropout(dropout)(x_final)
    x_final = layers.Dense(dense_units//2, activation="relu")(x_final)
    out = layers.Dense(1, activation="linear", name="yield_output")(x_final)

    model = models.Model(inputs=[input_weather, input_soil, input_mgmt], outputs=out, name="CNN_Transformer_LSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    print("\n Built CNN→Transformer→LSTM hybrid model.")
    model.summary()
    return model