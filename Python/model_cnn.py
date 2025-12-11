# src/hybrid_models.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_cnn_embedding_model(input_samples, input_channels, embed_dim=128, dropout=0.4):
    """
    Returns compiled Keras Model that maps (timesteps, channels) -> embedding vector.
    Classification head is optional (we'll keep a lightweight head for debugging, but we'll extract embedding from 'embedding' layer).
    """
    inp = layers.Input(shape=(input_samples, input_channels), name="epoch_input")
    x = inp

    # example conv stack (tune filter sizes/strides)
    x = layers.Conv1D(64, kernel_size=7, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(dropout)(x)
    embedding = layers.Dense(embed_dim, activation=None, name="embedding")(x)  # linear embedding
    x = layers.Activation("relu")(embedding)

    # optional small classification head for auxiliary loss (not required)
    out = layers.Dense(5, activation="softmax", name="softmax")(x)

    model = models.Model(inputs=inp, outputs=[out, embedding])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss={"softmax":"sparse_categorical_crossentropy"}, metrics={"softmax":"accuracy"})
    return model
