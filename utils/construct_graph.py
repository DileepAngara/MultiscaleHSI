import numpy as np

import torch

import networkx as nx
from skimage.segmentation import find_boundaries

def construct_graph(segments):
  unique_labels = np.unique(segments)

  graph = nx.Graph()

  graph.add_nodes_from(unique_labels)

  for label in unique_labels:
      neighbors = find_boundaries(segments == label, connectivity=1)
      neighbor_labels = np.unique(segments[neighbors])
      neighbor_labels = neighbor_labels[neighbor_labels != label]  # Remove the current superpixel label
      for neighbor_label in neighbor_labels:
          graph.add_edge(label, neighbor_label)
          graph.add_edge(neighbor_label, label)

  return torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous(), graph
