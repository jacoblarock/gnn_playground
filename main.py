from os.path import isfile
from os import makedirs
import pickle
from loader import load_graphs
from processor import *
from models import *

def mutag():
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
        df = neighbor_counts(df)
        df = l1_neighbor_counts(df)
        df = graph_weights(df)
        with open("data/processed/mutag.pkl", "wb") as file:
            pickle.dump(df, file)
    print(df)
    model = create_mlp_binary(7, 8, 1.5, 0.0)
    print(model.summary())
    fit(model, df, 200)
    df["prediction"] = predict(model, df)
    print(df)
    df.to_csv("out.csv")

def reddit():
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
        df = neighbor_counts(df)
        df = l1_neighbor_counts(df)
        df = graph_weights(df)
        with open("data/processed/reddit.pkl", "wb") as file:
            pickle.dump(df, file)
    print(df)
    model = create_mlp_binary(5, 8, 1.5, 0.0)
    print(model.summary())
    fit(model, df, 200)
    df["prediction"] = predict(model, df)
    print(df)
    df.to_csv("out.csv")

if __name__ == "__main__":
    reddit()