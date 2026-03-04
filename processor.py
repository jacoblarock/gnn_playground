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

def convert_to_undirected(df: pd.DataFrame) -> pd.DataFrame: 
    flipped = df.copy()
    flipped = flipped.rename({"a": "b", "b": "a"}, axis=1)
    df = pd.concat((df, flipped)).drop_duplicates().reset_index(drop=True)
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
    nodes = list(set(df["a"]))
    with mp.Pool(10) as pool:
        counts = pool.map(partial(_count_neighbors, df), nodes)
    temp = pd.DataFrame({
        "neighbors": counts
    }, index=nodes)
    df = df.join(temp, on="a")
    print()
    return df

def l1_neighbor_counts(df: pd.DataFrame) -> pd.DataFrame:
    for c in range(df["neighbors"].min(), df["neighbors"].max()+1):
        print(c)
        df[f"l1c{c}"] = 0
        for node in set(df.loc[df["neighbors"] == c,"b"]):
            print("node:", node, end="\r")
            df.loc[df["a"] == node,f"l1c{c}"] = df.loc[df["b"] == node].loc[df["neighbors"] == c].shape[0]
        print()
    return df

def _count_l1(df: pd.DataFrame, c: int, node: int) -> int:
    print("process node:", node, end="\r")
    return df.loc[df["b"] == node].loc[df["neighbors"] == c].shape[0]

def l1_neighbor_counts_mt(df: pd.DataFrame) -> pd.DataFrame:
    mapped: list[pd.Series] = []
    for c in range(df["neighbors"].min(), df["neighbors"].max()+1):
        nodes = list(set(df.loc[df["neighbors"] == c,"b"]))
        if len(nodes) > 0:
            print(c)
            if len(nodes) > 50:
                with mp.Pool(10) as pool:
                    counts = list(pool.map(partial(_count_l1, df, c), nodes))
            else:
                counts = list(map(partial(_count_l1, df, c), nodes))
            mapped.append(pd.Series(
                counts, index=nodes, name=f"l1c{c}"
            ))
            print()
        else:
            print("skip", c)
    for counts in mapped:
        print("join:", counts.name, end="\r")
        df = df.join(counts, on="a", how="left")
    print()
    return df

def graph_metadata(df: pd.DataFrame) -> pd.DataFrame:
    x_cols = list(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    data = pd.DataFrame(columns=[f"{col}_meta" for col in x_cols])
    for graph in set(df["graph"]):
        print("graph meta:", graph, end="\r")
        i_graph = df.loc[df["graph"] == graph].index
        for col in x_cols:
            data.loc[graph,f"{col}_meta"] = df.loc[i_graph,col].mean() # type: ignore
    print()
    df = df.join(data, on="graph")
    return df