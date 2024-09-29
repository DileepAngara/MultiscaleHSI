from utils.construct_graph import construct_graph
from utils.feature_initialization import (
    generate_features,
    superpixel_classes,
    generate_weights,
)
from utils.train_test_hsi import train_test_hsi
from utils.knn_graph import knn_graph
from torch_geometric.data import Data


def construct_feature_graph(segments, dataset, ground_truth, train_size, seed, beta, sigma_s, knn_k, k):
  edge_index, graph = construct_graph(segments) # Graph construction (in: segmentation label, out: COO graph)

  mean_features, weighted_features, centroids = generate_features(segments, dataset, k) # Feature initialization

  train_mask, test_mask = train_test_hsi(ground_truth, train_size, seed) # Train-test masking

  y, label_mask = superpixel_classes(segments, ground_truth, train_mask) # Initialize node labels

  edge_attr = generate_weights(mean_features, # Edge weights
                              weighted_features,
                              centroids, edge_index, beta, sigma_s)

  edge_index_knn, edge_attr_knn = knn_graph(edge_index, edge_attr, weighted_features, knn_k) # Create KNN graph

  data = Data(x=weighted_features, edge_index=edge_index_knn, y=y-1,
              edge_attr=edge_attr_knn, train_mask=label_mask,
              test_mask=test_mask, one_index=True)

  return data
