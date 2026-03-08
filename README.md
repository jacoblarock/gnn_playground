# gnn_playground
This Repository is a playground for certain methods of graph-level anomoly classification,
including data preprocessing on the node and graph level as well as various functions for model
creation and evaluation.

These experiments depend on tensorflow/keras, pandas and matplotlib (for training and evaluation
plotting).

Experiments were performend and tested on an AMD RX 9070 XT using tensorflow-rocm and with the 
reddit-binary and mutag datasets, which offer binary classifications based on graph-level
anomalousness.

# Training with Graph Embeddings
The embedding model is a node-level MLP that takes an input vector consisting of dimensions for
both graph-level and node-level features and maps them to a learned embedding-vector. The
graph-level embedding is produced from an aggregation of the node-level embeddings.

Training for these experiments is run with both negative and positive samples with one of several
loss functions that are either distance-based and conditional based on the true label of the graph
or based on the binary crossentropy of the sigmoid of the distance to the origin.

# Hypersphere Classification
Once a model is trained, a hypersphere with a dimensionality corresponding to the dimensionality
of the embedding space can be placed in the embeddings of the training data such that as many true
zero labels are contained within the hypersphere while excluding as many one-labels from the
hypersphere as possible, creating an ideal threshold. Classification of unseen datapoints can then
be done by embedding the graph and evaluating if the embedding is contained within this hypersphere.

# Classification with a Terminus Model
With a trained model, the embeddings of the training dataset on the graph level can then be used to
train a further model (MLP) to classify graphs based on the embeddings, potentially allowing for
more complex shapes than the hypersphere to differentiate embeddings.