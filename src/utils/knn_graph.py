from torch_geometric.utils import to_undirected
from sklearn.neighbors import NearestNeighbors
import torch


def knn_graph(edge_index, edge_attr, x, k):
    # Ensure edge_index is undirected
    edge_index = to_undirected(edge_index)

    if x.size(0) <= k:
        return edge_index, edge_attr

    # Convert to numpy array
    x_np = x.cpu().numpy()

    # Use sklearn's NearestNeighbors to find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(x_np)
    distances, indices = nbrs.kneighbors(x_np)

    # Convert indices to PyTorch tensor
    indices = torch.tensor(indices, dtype=torch.long).to(x.device)

    # Initialize edge lists
    edge_index_knn = []
    edge_attr_knn = []

    # Iterate through each node
    for i in range(indices.shape[0]):
        # Add edges and corresponding attributes to the edge lists
        for j in range(1, k):  # Skip the first index (self-loop)
            neighbor_idx = indices[i, j]
            edge_index_knn.append([i, neighbor_idx])
            edge_attr_knn.append(edge_attr[i])  # Assuming edge_attr is node-wise

    # Convert edge lists to PyTorch tensors
    edge_index_knn = torch.tensor(edge_index_knn, dtype=torch.long).t().contiguous()
    edge_attr_knn = torch.tensor(edge_attr_knn, dtype=edge_attr.dtype)

    return edge_index_knn, edge_attr_knn
