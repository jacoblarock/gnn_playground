import tensorflow as tf
from keras import models, layers, losses
import pandas as pd
import numpy as np

class CondSquareDistLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon=1e-7, name="custom_sum_loss"):
        super().__init__(name=name)
        # epsilon adjustment to gradient explosion
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        square_dist = tf.reduce_sum(tf.abs(y_pred), axis=-1)
        loss_if_zero = square_dist
        loss_if_one = 1.0 / (square_dist + self.epsilon)
        loss = tf.where(tf.cast(y_true, tf.bool), loss_if_one, loss_if_zero)
        return tf.reduce_mean(loss)

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
    )
    return model

def create_mlp_embedding(
    in_size: int,
    out_size: int,
    n_hl: int,
    hl_scale: float,
    hl_decay: float
) -> models.Sequential:
    model = models.Sequential([
        layers.Input((in_size,))
    ] + [
        layers.Dense(int(in_size * hl_scale * (1 - hl_decay * i)), activation="relu") for i in range(n_hl)
    ] + [
        layers.Dense(out_size, activation="linear")
    ])
    model.compile(
        optimizer="adam",
        loss=CondSquareDistLoss,
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
    model.fit(x, y, sample_weight=w, epochs=epochs)
    return model

def predict(
    model: models.Sequential,
    df: pd.DataFrame,
) -> np.ndarray:
    x_cols = list(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight", "prediction"})
    x = df[x_cols].to_numpy()
    return model.predict(x)

def graph_eval_linear(
    model: models.Sequential,
    df : pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(columns=["pred","true"])
    node_preds = predict(model, df)
    if len(node_preds.shape) > 0:
            node_preds = np.linalg.norm(node_preds, axis=1)
    for g in set(df["graph"]):
        out.loc[g] = {
            "pred": node_preds[list(df.loc[df["graph"] == g].index)].mean(),
            "true": df.loc[df["graph"] == g,"graph_label"].mean(), # type: ignore
        }
    return out