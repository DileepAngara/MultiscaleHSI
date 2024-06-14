import networkx as nx
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import numpy as np
import os
import torch

def construct_graph(segments, verbal=False, out=None):
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

  if verbal:
    print("Graph Construction:")
    print("==============================")
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    nx.draw(graph, node_size=10)
    plt.title('Graph Representation of Superpixel Segmentation')
    filepath = os.path.join(out, "superpixel_graph.png") if out else "superpixel_graph.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

  return torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous(), graph