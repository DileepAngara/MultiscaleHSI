from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
import os

def knn_graph_visualization(model, data, out):
    # Visualize the graph
    X = data.x.cpu()
    model.eval()
    with torch.no_grad():
        logits = model(data)
        labels = logits.argmax(dim=1).cpu() + 1

    # TSNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    G = to_networkx(data, to_undirected=True)
    pos = {node: coords for node, coords in zip(G.nodes(), X_tsne)}
    node_colors = labels

    fig, ax = plt.subplots()

    # Draw the graph with node colors based on labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.jet, node_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Create a legend
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color=plt.cm.jet(label / max(unique_labels)),
                        linestyle='', markersize=10, label=f'class {label}') for label in unique_labels]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

    ax.set_title('Graph with Node Colors Based on y Value')
    plt.axis('on')  # Turns on the axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    filepath = os.path.join(out, "knn_graph_visualization.png") if out else "knn_graph_visualization.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()