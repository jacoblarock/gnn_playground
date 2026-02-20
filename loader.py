import pandas as pd

def load_part(
    dataset: str,
    path: str,
    cols: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(
        f"data/{dataset}/{path}",
        delimiter=", ",
        header=None,
    )
    df.columns = cols
    return df

def merge_part(
    df: pd.DataFrame,
    dataset: str,
    path: str,
    cols: list[str],
    join_col: str | None,
) -> pd.DataFrame:
    part = load_part(dataset, path, cols)
    if join_col:
        part.index += 1
        return df.join(
            part,
            on=join_col
        ).reset_index(drop=True)
    df[cols] = part
    return df

def load_graphs(
    dataset: str,
    edge_path: str,
    graph_indic_path: str,
    graph_label_path: str,
    edge_label_path: str | None = None,
    vertex_label_path: str | None = None,
) -> pd.DataFrame:
    df = load_part(dataset, edge_path, ["a", "b"])
    df = merge_part(df, dataset, graph_indic_path, ["graph"], "a")
    df = merge_part(df, dataset, graph_label_path, ["graph_label"], "graph")
    df["graph_label"] = df["graph_label"].apply(
        lambda x : max(0, x)
    )
    if edge_label_path:
        df = merge_part(df, dataset, edge_label_path, ["edge_label"], None)
    if vertex_label_path:
        df = merge_part(df, dataset, vertex_label_path, ["vertex_label"], "a")
    return df
