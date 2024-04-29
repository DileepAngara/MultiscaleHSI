import networkx as nx
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import numpy as np

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

  print("Number of nodes:", graph.number_of_nodes())
  print("Number of edges:", graph.number_of_edges())
  plt.figure(figsize=(8, 8))
  nx.draw(graph, node_size=10)
  plt.title('Graph Representation of Superpixel Segmentation')
  plt.show()

  return graph