import tensorflow as tf
from keras import models, layers, losses, metrics
import pandas as pd
import numpy as np

def create_mlp_binary(
    in_size: int,
    n_hl: int,
    hl_scale: float,
    hl_decay: float
) -> models.Sequential:
    model = models.Sequential([
        layers.Input((in_size,))
    ] + [
        layers.Dense(int(in_size * hl_scale * (1 - hl_decay * i)), activation="relu") for i in range(n_hl)
    ] + [
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss=losses.BinaryCrossentropy,
        metrics=[metrics.Accuracy]
    )
    return model

def fit(
    model: models.Sequential,
    df: pd.DataFrame,
    epochs: int,
) -> models.Sequential:
    x_cols = list(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    x = df[x_cols].to_numpy()
    y = df["graph_label"].to_numpy()
    w = df["graph_weight"].to_numpy()
    model.fit(x, y, sample_weight=w, epochs=epochs, validation_split=0.2)
    return model

def predict(
    model: models.Sequential,
    df: pd.DataFrame,
) -> np.ndarray:
    x_cols = list(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    x = df[x_cols].to_numpy()
    return model.predict(x)