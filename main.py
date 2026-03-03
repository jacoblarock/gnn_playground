from os.path import isfile
from os import makedirs
import pickle
from loader import load_graphs
from processor import *
from models import *
from datetime import datetime

def load_mutag() -> pd.DataFrame:
    makedirs("data/processed", exist_ok=True)
    cache_path = f"data/processed/mutag.csv"
    if not isfile(cache_path):
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
        df.to_csv(cache_path, index=False)
    df = pd.read_csv(cache_path)
    return df

def load_reddit() -> pd.DataFrame:
    cutoff = 100
    cache_path = f"data/processed/reddit{cutoff}.csv"
    makedirs("data/processed", exist_ok=True)
    if not isfile(cache_path):
        df = load_graphs(
            dataset="REDDIT-BINARY",
            edge_path="REDDIT-BINARY_A.txt",
            graph_indic_path="REDDIT-BINARY_graph_indicator.txt",
            graph_label_path="REDDIT-BINARY_graph_labels.txt",
        )
        graphs_false = list(set(df.loc[df["graph_label"]==0,"graph"]))[:cutoff]
        graphs_true = list(set(df.loc[df["graph_label"]==1,"graph"]))[:cutoff]
        graphs = graphs_false + graphs_true
        print(graphs)
        print(df.shape)
        df = df.loc[df["graph"].isin(graphs)]
        print(df.shape)
        df = neighbor_counts_mt(df)
        print(set(df["neighbors"]))
        df = l1_neighbor_counts_mt(df).fillna(0).astype(int)
        df = graph_metadata(df)
        df = graph_weights(df)
        df.to_csv(cache_path, index=False)
    df = pd.read_csv(cache_path)
    return df

def exp_embedding_hypersphere():
    # loading and training
    train_split = 0.6
    makedirs("out", exist_ok=True)
    df = load_reddit()
    print(df)
    print(df["graph"].max())
    graphs_false = list(set(df.loc[df["graph_label"]==0,"graph"]))
    false_cutoff = int(len(graphs_false) * train_split)
    graphs_true = list(set(df.loc[df["graph_label"]==1,"graph"]))
    true_cutoff = int(len(graphs_true) * train_split)
    train = pd.concat((
        df.loc[df["graph"].isin(graphs_false[:false_cutoff])].reset_index(drop=True),
        df.loc[df["graph"].isin(graphs_true[:true_cutoff])].reset_index(drop=True),
    )).reset_index(drop=True)
    test = pd.concat((
        df.loc[df["graph"].isin(graphs_false[false_cutoff:])].reset_index(drop=True),
        df.loc[df["graph"].isin(graphs_true[true_cutoff:])].reset_index(drop=True),
    )).reset_index(drop=True)
    in_size = len(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    model = create_mlp_embedding(in_size, 14, 8, 1.5, 0.1)
    print(model.summary())
    model = fit(model, train, 10)
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
    graphs_false = list(set(df.loc[df["graph_label"]==0,"graph"]))
    false_cutoff = int(len(graphs_false) * train_split)
    graphs_true = list(set(df.loc[df["graph_label"]==1,"graph"]))
    true_cutoff = int(len(graphs_true) * train_split)
    train = pd.concat((
        df.loc[df["graph"].isin(graphs_false[:false_cutoff])].reset_index(drop=True),
        df.loc[df["graph"].isin(graphs_true[:true_cutoff])].reset_index(drop=True),
    )).reset_index(drop=True)
    test = pd.concat((
        df.loc[df["graph"].isin(graphs_false[false_cutoff:])].reset_index(drop=True),
        df.loc[df["graph"].isin(graphs_true[true_cutoff:])].reset_index(drop=True),
    )).reset_index(drop=True)
    in_size = len(set(df.columns) - {"a", "b", "graph", "graph_label", "graph_weight"})
    model = create_mlp_embedding(in_size, 14, 8, 1.5, 0.1)
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
    exp_embedding_hypersphere()