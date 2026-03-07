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
        df = df.loc[df["graph"].isin(graphs)]
        # df = convert_to_undirected(df)
        df = neighbor_counts_mt(df)
        print(set(df["neighbors"]))
        df = l1_neighbor_counts_mt(df).fillna(0).astype(int)
        df = graph_metadata(df)
        df = graph_weights(df)
        df.to_csv(cache_path, index=False)
    df = pd.read_csv(cache_path)
    return df

def exp_embedding_hypersphere(
    train: pd.DataFrame,
    test: pd.DataFrame,
    e_in_size: int,
    e_out_size: int,
    e_n_hl: int,
    e_hl_scale: float,
    e_hl_decay: float,
    e_epochs: int,
) -> dict:
    model = create_mlp_embedding(
        e_in_size,
        e_out_size,
        e_n_hl,
        e_hl_scale,
        e_hl_decay
    )
    print(model.summary())
    model = fit(model, train, e_epochs)
    threshold = best_threshold(model, train)
    print("threshold:", threshold)
    # testing
    g_eval = graph_eval(model, test)
    # g_eval.to_csv("out/eval.csv", index=False)
    return eval_stats(g_eval, threshold)

def batch_eval_hs(
    *args,
    exp = exp_embedding_hypersphere,
    n_batches=1
) -> dict:
    results = []
    for i in range(n_batches):
        results.append(exp(*args))
    df = pd.DataFrame(results)
    out = {}
    for col in df.columns:
        out[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
        }
    return out


def exp_terminus(
    train: pd.DataFrame,
    test: pd.DataFrame,
    e_in_size: int,
    e_out_size: int,
    e_n_hl: int,
    e_hl_scale: float,
    e_hl_decay: float,
    e_epochs: int,
) -> dict:
    model = create_mlp_embedding(
        e_in_size,
        e_out_size,
        e_n_hl,
        e_hl_scale,
        e_hl_decay
    )
    print(model.summary())
    model = fit(model, train, e_epochs)
    # train terminus
    term = create_mlp_terminus(model, 2, 1.2, 0.0)
    print(term.summary())
    term_x = predict(model, train)
    term = fit_terminus(term, train, term_x, 200)
    return eval_term(term, test, term_x)

def experiment():
    train_split = 0.6
    makedirs("out", exist_ok=True)
    df = load_mutag()
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
    res = batch_eval_hs(
        train,
        test,
        in_size,
        14,
        8,
        1.2,
        0.0,
        200,
        n_batches=2
    )

if __name__ == "__main__":
    experiment()