# src/hybrid_utils.py
import numpy as np
from tensorflow.keras.models import Model

def get_embedding_model(full_model):
    """
    Given model with named layer 'embedding' returns a model that outputs embeddings.
    """
    emb_layer = full_model.get_layer("embedding").output
    emb_model = Model(inputs=full_model.input, outputs=emb_layer)
    return emb_model

def extract_embeddings(emb_model, X, batch_size=64):
    """
    X shape expected: (n_epochs, samples, channels)
    returns (n_epochs, embed_dim)
    """
    return emb_model.predict(X, batch_size=batch_size, verbose=1)
