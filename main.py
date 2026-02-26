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

def load_reddit(test=False) -> pd.DataFrame:
    cutoff = 20
    cache_path = f"data/processed/reddit{cutoff}.pkl" if not test else f"data/processed/reddit{cutoff}_test.pkl"
    makedirs("data/processed", exist_ok=True)
    if isfile(cache_path):
        with open(cache_path, "rb") as file:
            df = pickle.load(file)
    else:
        df = load_graphs(
            dataset="REDDIT-BINARY",
            edge_path="REDDIT-BINARY_A.txt",
            graph_indic_path="REDDIT-BINARY_graph_indicator.txt",
            graph_label_path="REDDIT-BINARY_graph_labels.txt",
        )
        if not test:
            graphs_false = list(set(df.loc[df["graph_label"]==0,"graph"]))[:cutoff]
            graphs_true = list(set(df.loc[df["graph_label"]==1,"graph"]))[:cutoff]
        else:
            graphs_false = list(set(df.loc[df["graph_label"]==0,"graph"]))[-cutoff:]
            graphs_true = list(set(df.loc[df["graph_label"]==1,"graph"]))[-cutoff:]
        graphs = graphs_false + graphs_true
        print(df.shape)
        df = df.loc[df["graph"].isin(graphs)]
        print(df.shape)
        df = neighbor_counts_mt(df)
        print(set(df["neighbors"]))
        df = l1_neighbor_counts_mt(df).fillna(0).astype(int)
        df = graph_metadata(df)
        df = graph_weights(df)
        with open(cache_path, "wb") as file:
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

def exp_terminus_mutag():
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
    model = create_mlp_embedding(in_size, 10, 8, 1.5, 0.0)
    print(model.summary())
    model = fit(model, train, 200)
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

def exp_terminus_reddit():
    # loading and training
    makedirs("out", exist_ok=True)
    train = load_reddit()
    print(train)
    print(train["graph"].max())
    in_size = len(set(train.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    model = create_mlp_embedding(in_size, 10, 8, 0.8, 0.1)
    print(model.summary())
    model = fit(model, train, 1)
    # train terminus
    term = create_mlp_terminus(model, 2, 1.2, 0.0)
    print(term.summary())
    term_x = predict(model, train)
    term = fit_terminus(term, train, term_x, 200)
    del train
    del term_x
    test = load_reddit(test=True)
    term_x = predict(model, test)
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
    exp_terminus_mutag()