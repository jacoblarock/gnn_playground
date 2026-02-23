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
        df = l1_neighbor_counts_mt(df).fillna(0).astype(int)
        df = graph_metadata(df)
        df = graph_weights(df)
        with open("data/processed/mutag.pkl", "wb") as file:
            pickle.dump(df, file)
    return df

def load_reddit() -> pd.DataFrame:
    cutoff = 50
    makedirs("data/processed", exist_ok=True)
    if isfile(f"data/processed/reddit{cutoff}.pkl"):
        with open(f"data/processed/reddit{cutoff}.pkl", "rb") as file:
            df = pickle.load(file)
    else:
        df = load_graphs(
            dataset="REDDIT-BINARY",
            edge_path="REDDIT-BINARY_A.txt",
            graph_indic_path="REDDIT-BINARY_graph_indicator.txt",
            graph_label_path="REDDIT-BINARY_graph_labels.txt",
        )
        graphs = list(set(df["graph"]))[0:cutoff]
        print(df.shape)
        df = df.loc[df["graph"].isin(graphs)]
        print(df.shape)
        df = neighbor_counts_mt(df)
        print(set(df["neighbors"]))
        df = l1_neighbor_counts_mt(df).fillna(0).astype(int)
        df = graph_metadata(df)
        df = graph_weights(df)
        with open(f"data/processed/reddit{cutoff}.pkl", "wb") as file:
            pickle.dump(df, file)
    return df

def exp_embedding_dist():
    # loading and training
    train_split = 0.6
    makedirs("out", exist_ok=True)
    df = load_mutag()
    print(df)
    print(df["graph"].max())
    split_point = int(df["graph"].max() * train_split)
    train = df.loc[df["graph"] <= split_point].reset_index(drop=True)
    test = df.loc[df["graph"] > split_point].reset_index(drop=True)
    in_size = len(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    model = create_mlp_embedding(in_size, 14, 8, 1.5, 0.0)
    print(model.summary())
    model = fit(model, train, 200)
    threshold = best_threshold(model, train)
    print("threshold:", threshold)
    # testing
    g_eval = graph_eval(model, test)
    stats = eval_stats(g_eval, threshold)
    for key, val in stats.items():
        print(key, val)

def exp_terminus():
    # loading and training
    train_split = 0.6
    makedirs("out", exist_ok=True)
    df = load_reddit()
    print(df)
    print(df["graph"].max())
    split_point = int(df["graph"].max() * train_split)
    train = df.loc[df["graph"] <= split_point].reset_index(drop=True)
    test = df.loc[df["graph"] > split_point].reset_index(drop=True)
    in_size = len(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    model = create_mlp_embedding(in_size, 10, 8, 1.5, 0.0)
    print(model.summary())
    model = fit(model, train, 10)
    # train terminus
    term = create_mlp_terminus(model, 2, 1.2, 0.0)
    print(term.summary())
    term_x = predict(model, train)
    term = fit_terminus(term, train, term_x, 200)
    t_eval = eval_term(term, test, term_x)
    eval_df = pd.DataFrame({
        "metric": [
            "Loss",
            "Accuracy",
            "Precision",
            "Recall",
            "TruePositives",
            "TrueNegatives",
            "FalsePositives",
            "FalseNegatives",
        ],
        "value": t_eval,
    })
    eval_df = eval_df.loc[eval_df["metric"] != "loss"].reset_index(drop=True)
    print(eval_df)

if __name__ == "__main__":
    exp_terminus()