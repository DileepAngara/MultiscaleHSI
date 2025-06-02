import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

from config import (
    DATA_PATH,
    RESULTS_PATH,
    SEED,
    TRAIN_SIZE,

    SIGMA_S,
    KNN_K,
    K,
    BETA, 

    NHID, 
    DROPOUT, 
    EPOCH,
)
from utils.load_hsi import load_hsi, seed_everything
from utils.segmentation import get_false_color, segmentation
from utils.find_pca import find_pca
from utils.visualization import (
    visualize_dataset, 
    visualize_segmentation, 
    visualize_graph, 
    plot_training_results, 
    visualize_cmap
)
from utils.construct_feature_graph import construct_feature_graph
from utils.construct_graph import construct_graph
from utils.graph_loss import GraphLoss
from utils.training_loop import train, test, get_cmap

import torch
from models import MGNN
from sklearn.manifold import TSNE
import networkx as nx
import torch_geometric

# Create directories if they do not exist
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

# Setup logging
def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),  # Log to file
                            logging.StreamHandler(sys.stdout)  # Log to console
                        ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Select INDIAN, SALINAS, PAVIA, KENNEDY, BOTSWANA, TORONTO",
    )
    parser.add_argument(
        "--segmentation_size",
        type=int,
        help="Segmentation Size (10, 100, 200, 200, 100, 200)",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Train? (Y or N)",
    )

    args = parser.parse_args()

    DATASET = args.dataset
    SEGMENTATION_SIZE = args.segmentation_size

    seed_everything(SEED)

    # Create dataset-specific directories
    DATASET_RESULT_PATH = os.path.join(RESULTS_PATH, DATASET)
    os.makedirs(DATASET_RESULT_PATH, exist_ok=True)

    EXPERIMENT_RESULT_PATH = os.path.join(DATASET_RESULT_PATH, "experiment")
    os.makedirs(EXPERIMENT_RESULT_PATH, exist_ok=True)

    # Setup logging with file output
    log_file = os.path.join(EXPERIMENT_RESULT_PATH, "log.txt")
    setup_logging(log_file)

    logging.info(f"Processing model: MOB-GCN")
    logging.info(f"Processing dataset: {DATASET}")
    
    # Load dataset
    dataset, ground_truth = load_hsi(DATASET, DATA_PATH)
    false_color = get_false_color(dataset)

    dataset_pca = find_pca(dataset, 0.999)

    visualize_dataset(dataset, ground_truth, false_color, os.path.join(EXPERIMENT_RESULT_PATH, "visualize_dataset.png"))

    NFEAT = dataset_pca.shape[2]
    NOUT = len(np.unique(ground_truth[ground_truth!=0]))

    # Segmentation
    segments = segmentation(dataset, SEGMENTATION_SIZE)
    visualize_segmentation(segments, false_color, ground_truth, os.path.join(EXPERIMENT_RESULT_PATH, "visualize_segmentation.png"))

    edge_index, graph = construct_graph(segments)
    visualize_graph(graph, os.path.join(EXPERIMENT_RESULT_PATH, "visualize_graph.png"))

    # Construct feature graph
    data = construct_feature_graph(segments, dataset_pca, ground_truth, TRAIN_SIZE, SEED, BETA, SIGMA_S, KNN_K, K)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_CLUSTERS = [NOUT]

    model = MGNN(nfeat = NFEAT,
              nhid = NHID,
              nout = NOUT,
              dropout = DROPOUT, num_clusters = NUM_CLUSTERS).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())

    loss_history, acc_history = [], []

    if args.training:
        criterion = GraphLoss()

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for epoch in range(EPOCH+1):
            loss = train(model, device, optimizer, criterion, data)
            acc, _, _, _, _ = test(model, device, segments, ground_truth, data)

            loss_history.append(loss)
            acc_history.append(acc)

            if epoch % 50 == 0:
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        logging.info(f'Saving model to {os.path.join(EXPERIMENT_RESULT_PATH, "mgn_model.pth")}')
        torch.save({
                  "model_state_dict": model.state_dict(),
                  "num_clusters": NUM_CLUSTERS
        },os.path.join(EXPERIMENT_RESULT_PATH, "mgn_model.pth"))
        torch.save(optimizer.state_dict(), os.path.join(EXPERIMENT_RESULT_PATH, "mgn_optimizer.pth"))
        np.save(os.path.join(EXPERIMENT_RESULT_PATH, "mgn_loss_history.npy"), loss_history)
        np.save(os.path.join(EXPERIMENT_RESULT_PATH, "mgn_acc_history.npy"), acc_history)
    else:
        logging.info(f'Loading model from {os.path.join(EXPERIMENT_RESULT_PATH, "mgn_model.pth")}')
        model.load_state_dict(torch.load(os.path.join(EXPERIMENT_RESULT_PATH, "mgn_model.pth"), weights_only=True))
        optimizer.load_state_dict(torch.load(os.path.join(EXPERIMENT_RESULT_PATH, "mgn_optimizer.pth"), weights_only=True))
        loss_history = np.load(os.path.join(EXPERIMENT_RESULT_PATH, "mgn_loss_history.npy"))
        acc_history = np.load(os.path.join(EXPERIMENT_RESULT_PATH, "mgn_acc_history.npy"))

        for epoch in range(0, EPOCH+1, 50):
            loss = loss_history[epoch]
            acc = acc_history[epoch]
            logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

    oa, aa, ka, _, _ = test(model, device, segments, ground_truth, data)
    logging.info(f'OA: {oa:.4f}, AA: {aa:.4f}, KA: {ka:.4f}')

    cmap = get_cmap(model, device, segments, data)
    plot_training_results(EPOCH, loss_history, acc_history, os.path.join(EXPERIMENT_RESULT_PATH, "mgn_training_loss.png"))
    visualize_cmap(cmap, ground_truth, os.path.join(EXPERIMENT_RESULT_PATH, "mgn_visualize_cmap.png"))

    # TSNE and graph visualizations
    # Extract features and model logits
    X = data.x.cpu()
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        labels = logits.argmax(dim=1).cpu() + 1

    # TSNE visualization for spatial embeddings
    tsne_spatial = TSNE(n_components=2, random_state=SEED)
    X_tsne_spatial = tsne_spatial.fit_transform(X)

    # TSNE visualization for spectral embeddings (logits)
    tsne_spectral = TSNE(n_components=2, random_state=SEED)
    X_tsne_spectral = tsne_spectral.fit_transform(logits.cpu())

    # Get the color values for each label
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    pred_labels = data.y.cpu() + 1

    # Scatter plot marker size and alpha
    scatter_size = 40
    scatter_alpha = 0.7

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # Plot for spatial embeddings
    for label, color in zip(unique_labels, colors):
        indices = np.where(np.array(pred_labels) == label)
        ax[0].scatter(X_tsne_spatial[indices, 0], X_tsne_spatial[indices, 1],
                    c=[color], label=f'Label {label}', s=scatter_size, alpha=scatter_alpha, edgecolors='k', linewidth=0.5)

    ax[0].set_title("TSNE Visualization of Spatial Features", fontsize=16)
    ax[0].set_xlabel("Component 1", fontsize=12)
    ax[0].set_ylabel("Component 2", fontsize=12)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].tick_params(axis='both', which='major', labelsize=10)

    # Plot for spectral embeddings (logits)
    for label, color in zip(unique_labels, colors):
        indices = np.where(np.array(labels) == label)
        ax[1].scatter(X_tsne_spectral[indices, 0], X_tsne_spectral[indices, 1],
                    c=[color], label=f'Label {label}', s=scatter_size, alpha=scatter_alpha, edgecolors='k', linewidth=0.5)

    ax[1].set_title("TSNE Visualization of Spectral Embeddings", fontsize=16)
    ax[1].set_xlabel("Component 1", fontsize=12)
    ax[1].set_ylabel("Component 2", fontsize=12)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].tick_params(axis='both', which='major', labelsize=10)

    # Improved legend layout
    handles, legend_labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center right', fontsize=10, borderaxespad=0.1, bbox_to_anchor=(1.1, 0.5))
    plt.tight_layout()

    # Save as a higher resolution image
    output_path = os.path.join(EXPERIMENT_RESULT_PATH, "mgn_embeddings.png")
    plt.savefig(output_path, dpi=600)
    # plt.show()

    # Improved visualization of the graph with node colors based on class labels
    G = torch_geometric.utils.to_networkx(data, to_undirected=True)
    pos = {node: coords for node, coords in zip(G.nodes(), X_tsne_spatial)}
    node_colors = labels

    fig, ax = plt.subplots(figsize=(10, 8))  # Larger figure size for better visibility

    # Draw the graph with improved node and edge settings
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.get_cmap('jet'),
                                node_size=50, alpha=0.85, ax=ax, linewidths=0.5, edgecolors='k')
    edges = nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)  # Slight transparency and thin edges

    # Create a legend with distinct colors for each class
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color=plt.cm.get_cmap('jet')(label / max(unique_labels)),
                        linestyle='', markersize=10, label=f'Class {label}') for label in unique_labels]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title="Classes", fontsize=10)

    # Customize the appearance of the plot
    ax.set_title('Graph Visualization with Node Colors Based on Class Labels', fontsize=14)
    plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("Component 1 (TSNE)", fontsize=12)
    ax.set_ylabel("Component 2 (TSNE)", fontsize=12)

    # Save the figure in a different path
    output_path = os.path.join(EXPERIMENT_RESULT_PATH, "mgn_embedding_graph.png")
    plt.savefig(output_path, dpi=600)  # Save at a higher DPI for better resolution


if __name__ == "__main__":
    main()
