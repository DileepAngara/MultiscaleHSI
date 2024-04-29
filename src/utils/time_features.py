from tqdm import tqdm
import numpy as np
import torch
import networkx as nx
from skimage.segmentation import find_boundaries
from feature_initialization import calc_weight, superpixel_classes, generate_weights
from construct_graph import construct_graph
from train_test_hsi import train_test_hsi
from knn_graph import knn_graph
from torch_geometric.data import Data

def generate_feature_windows(segments, img, window, alpha=1):
  unique_labels = np.unique(segments)
  timeframe = img.shape[2] - window + 1

  mean_features = np.zeros((timeframe, len(unique_labels), window))
  centroids = []
  print("Generating Mean Window Features and Centroids")
  for label in tqdm(unique_labels):
    mask = (segments == label)

    rows, cols = np.where(mask)
    centroids.append((np.mean(rows), np.mean(cols)))

    for time in range(timeframe):
      masked_image = img[:,:,time:time+window] * np.expand_dims(mask, axis=-1)
      mean_features[time][label] = np.mean(masked_image, axis=(0, 1))

  weighted_features = np.zeros((timeframe, len(unique_labels), window))

  print("Generating Weighted Window Features")
  for label in tqdm(unique_labels):
    neighbors = find_boundaries(segments == label, connectivity=1)
    neighbor_labels = np.unique(segments[neighbors])

    for time in range(timeframe):
      neighbor_vectors = mean_features[time][neighbor_labels]
      mean_vector = mean_features[time][label]

      neighbor_weights = [(calc_weight(mean_vector, neighbor_vector, alpha)) for neighbor_vector in neighbor_vectors]
      neighbor_weights = np.array(neighbor_weights) / np.sum(neighbor_weights)
      if np.sum(neighbor_weights) > 0:
        weighted_features[time][label] = np.sum(mean_features[time][neighbor_labels] * neighbor_weights[...,np.newaxis], axis=0)
      else:
        weighted_features[time][label] = mean_vector # Weights arent strong enough then consider it as its own neighborhood

  return torch.tensor(mean_features, dtype=torch.float), torch.tensor(weighted_features, dtype=torch.float), torch.tensor(np.array(centroids), dtype=torch.float)

def construct_feature_time_graph(segments, dataset, ground_truth, WINDOW, TRAIN_SIZE, SEED, BETA, SIGMA_S, SIGMA_L, KNN_K, K):
  graph = construct_graph(segments) # Graph construction (in: segmentation label, out: COO graph)

  edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()

  train_mask, test_mask = train_test_hsi(ground_truth, TRAIN_SIZE, SEED) # Train-test masking

  y, label_mask = superpixel_classes(segments, ground_truth, train_mask) # Initialize node labels

  mean_features_windows, weighted_features_windows, centroids = generate_feature_windows(segments,
          dataset, WINDOW, K)

  timeframe_data = []
  for timeframe in tqdm(range(dataset.shape[2]-WINDOW+1)):
    mean_features= mean_features_windows[timeframe] # Feature initialization
    weighted_features = weighted_features_windows[timeframe]
    x = weighted_features

    edge_attr = generate_weights(mean_features, # Edge weights
                                weighted_features,
                                centroids, edge_index, BETA, SIGMA_S, SIGMA_L)

    edge_index_knn, edge_attr_knn = knn_graph(edge_index, edge_attr, x, KNN_K) # Create KNN graph
    data = Data(x=x, edge_index=edge_index_knn, y=y,
                edge_attr=edge_attr_knn, num_nodes=len(np.unique(segments)),
                train_mask = label_mask, test_mask=test_mask,
                one_index=True)

    timeframe_data.append(data)

  return timeframe_data