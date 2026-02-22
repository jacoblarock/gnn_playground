from os.path import isfile
from os import makedirs
import pickle
from loader import load_graphs
from processor import *
from models import *
from datetime import datetime

def load_mutag() -> pd.DataFrame:
    makedirs("data/processed", exist_ok=True)
    if isfile("data/processed/mutag.pkl"):
        with open("data/processed/mutag.pkl", "rb") as file:
            df = pickle.load(file)
    else:
        df = load_graphs(
            dataset="MUTAG",
            edge_path="MUTAG_A.txt",
            graph_indic_path="MUTAG_graph_indicator.txt",
            graph_label_path="MUTAG_graph_labels.txt",
            edge_label_path="MUTAG_edge_labels.txt",
            vertex_label_path="MUTAG_node_labels.txt",
        )
        df = neighbor_counts_mt(df)
        print(set(df["neighbors"]))
        df = l1_neighbor_counts_mt(df)
        df = graph_weights(df)
        with open("data/processed/mutag.pkl", "wb") as file:
            pickle.dump(df, file)
    return df

def load_reddit() -> pd.DataFrame:
    makedirs("data/processed", exist_ok=True)
    if isfile("data/processed/reddit.pkl"):
        with open("data/processed/reddit.pkl", "rb") as file:
            df = pickle.load(file)
    else:
        df = load_graphs(
            dataset="REDDIT-BINARY",
            edge_path="REDDIT-BINARY_A.txt",
            graph_indic_path="REDDIT-BINARY_graph_indicator.txt",
            graph_label_path="REDDIT-BINARY_graph_labels.txt",
        )
        graphs = []
        for graph in set(df["graph"]):
            graphs.append(
                df.loc[df["graph"] == graph].copy()
            )
        del df
        graphs = map(neighbor_counts_mt, graphs)
        graphs = map(l1_neighbor_counts_mt, graphs)
        df = pd.concat(graphs).fillna(0).astype(int)
        del graphs
        df = graph_weights(df)
        with open("data/processed/reddit.pkl", "wb") as file:
            pickle.dump(df, file)
    return df

def main():
    train_split = 0.8
    makedirs("out", exist_ok=True)
    df = load_reddit()
    print(df)
    print(df["graph"].max())
    split_point = int(df["graph"].max() * train_split)
    train = df.loc[df["graph"] <= split_point].reset_index(drop=True)
    test = df.loc[df["graph"] > split_point].reset_index(drop=True)
    # model = create_mlp_binary(7, 8, 1.5, 0.0)
    model = create_mlp_embedding(5, 5, 8, 1.5, 0.0)
    print(model.summary())
    model = fit(model, train, 400)
    g_eval = graph_eval_linear(model, test)
    res0 = g_eval.loc[g_eval["true"] == 0,"pred"]
    res1 = g_eval.loc[g_eval["true"] == 1,"pred"]
    print("Results 0 labels:")
    print("mean:", res0.mean())
    print("stdev:", res0.std())
    print("Results 1 labels:")
    print("mean:", res1.mean())
    print("stdev:", res1.std())

if __name__ == "__main__":
    main()