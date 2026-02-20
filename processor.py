import pandas as pd

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

def l1_neighbor_counts(df: pd.DataFrame) -> pd.DataFrame:
    for c in range(df["neighbors"].min(), df["neighbors"].max()+1):
        print(c)
        df[f"l1c{c}"] = 0
        for node in set(df["a"]):
            print("node:", node, end="\r")
            df.loc[df["a"] == node,f"l1c{c}"] = df.loc[df["b"] == node].loc[df["neighbors"] == c].shape[0]
        print()
    return df
