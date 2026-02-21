import pandas as pd
import multiprocessing as mp
from functools import partial
import numpy as np

def graph_weights(df: pd.DataFrame) -> pd.DataFrame:
    df["graph_weight"] = 0.0
    for g in set(df["graph"]):
        df.loc[df["graph"] == g,"graph_weight"] = 1 / df.loc[df["graph"] == g].shape[0]
    df["graph_weight"] /= df["graph_weight"].max()
    return df

def neighbor_counts(df: pd.DataFrame) -> pd.DataFrame:
    df["neighbors"] = 0
    for node in set(df["a"]):
        print("node:", node, end="\r")
        df.loc[df["a"] == node,"neighbors"] = df.loc[df["a"] == node].shape[0]
    print()
    return df

def _gen_batches(data: pd.Series, batch_size: int) -> list[np.ndarray]:
    out = []
    temp = np.array(list(set(data)))
    for i in range(0, len(temp), batch_size):
        out.append(temp[i:i+batch_size])
    return out

def _count_neighbors(df: pd.DataFrame, node: int) -> int:
    print("process node:", node, end="\r")
    return df.loc[df["a"] == node].shape[0]

def neighbor_counts_mt(df: pd.DataFrame) -> pd.DataFrame:
    df["neighbors"] = 0
    with mp.Pool(10) as pool:
        counts = pool.map(partial(_count_neighbors, df), set(df["a"]))
    print()
    for node, count in enumerate(counts):
        print("assign node:", node, end="\r")
        df.loc[df["a"] == node,"neighbors"] = count
    print()
    return df

def l1_neighbor_counts(df: pd.DataFrame) -> pd.DataFrame:
    for c in range(df["neighbors"].min(), df["neighbors"].max()+1):
        print(c)
        df[f"l1c{c}"] = 0
        for node in set(df.loc[df["neighbors"] == c,"a"]):
            print("node:", node, end="\r")
            df.loc[df["a"] == node,f"l1c{c}"] = df.loc[df["b"] == node].loc[df["neighbors"] == c].shape[0]
        print()
    return df

def _count_l1(df: pd.DataFrame, c: int, node: int):
    print("process node:", node, end="\r")
    return df.loc[df["b"] == node].loc[df["neighbors"] == c].shape[0]

def l1_neighbor_counts_mt(df: pd.DataFrame) -> pd.DataFrame:
    for c in range(df["neighbors"].min(), df["neighbors"].max()+1):
        print(c)
        df[f"l1c{c}"] = 0
        with mp.Pool(10) as pool:
            counts = pool.map(partial(_count_l1, df, c), set(df.loc[df["neighbors"] == c,"a"]))
        nodes = list(df.loc[df["neighbors"] == c,"a"])
        for i_node, count in enumerate(counts):
            node = nodes[i_node]
            print("assign node:", node, end="\r")
            df.loc[df["a"] == node,f"l1c{c}"] = count
        print()
    return df
