import os
import argparse
import numpy as np
import torch
from models import MGNN
from utils.load_hsi import load_hsi, seed_everything
from utils.segmentation import get_false_color, segmentation
from utils.find_pca import find_pca
from utils.visualization import visualize_dataset, visualize_segmentation, visualize_graph, visualize_cmap
from utils.construct_feature_graph import construct_feature_graph
from utils.construct_graph import construct_graph
from utils.training_loop import test, get_cmap
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

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Select INDIAN, SALINAS, PAVIA, KENNEDY, BOTSWANA, TORONTO")
    parser.add_argument("--segmentation_size", type=int, required=True, help="Segmentation Size (e.g. 10, 100, 200)")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output visualizations")
    parser.add_argument("--num_clusters", type=str, help="Comma-separated list of cluster numbers (e.g. '5,10,15')")
    args = parser.parse_args()

    # Ensure the output path exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Set random seed
    seed_everything(SEED)

    # Load dataset and perform PCA
    dataset, ground_truth = load_hsi(args.dataset, DATA_PATH)
    false_color = get_false_color(dataset)
    dataset_pca = find_pca(dataset, 0.999)

    # Visualize dataset
    visualize_dataset(dataset, ground_truth, false_color, os.path.join(args.output_path, "visualize_dataset.png"))

    # Perform segmentation
    segments = segmentation(dataset, args.segmentation_size)
    visualize_segmentation(segments, false_color, ground_truth, os.path.join(args.output_path, "visualize_segmentation.png"))

    # Construct the graph for inference
    edge_index, graph = construct_graph(segments)
    visualize_graph(graph, os.path.join(args.output_path, "visualize_graph.png"))

    data = construct_feature_graph(segments, dataset_pca, ground_truth, TRAIN_SIZE, SEED, BETA, SIGMA_S, KNN_K, K)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with the same architecture used for training
    NOUT = len(np.unique(ground_truth[ground_truth != 0]))
    
    '''if args.num_clusters:
        NUM_CLUSTERS = list(map(int, args.num_clusters.split(',')))
    else:
        NUM_CLUSTERS = [NOUT]

    model = MGNN(nfeat=data.num_node_features,
                nhid=NHID,
                nout=NOUT,
                dropout = DROPOUT, num_clusters = NUM_CLUSTERS).to(device)'''

    # Load checkpoint dictionary
    print(f"Loading model weights from {args.weights_path}")
    checkpoint = torch.load(args.weights_path)

    # Load num_clusters from file if not passed in args
    if args.num_clusters:
      NUM_CLUSTERS = list(map(int, args.num_clusters.split(',')))
    else:
      NUM_CLUSTERS = checkpoint["num_clusters"]

    # Rebuild model using correct architecture
    model = MGNN(
            nfeat=data.num_node_features,
            nhid=NHID,
            nout=NOUT,
            dropout=DROPOUT,
            num_clusters=NUM_CLUSTERS
    ).to(device)

    # Now load the weights safely
    model.load_state_dict(checkpoint["model_state_dict"])

    # Perform inference
    model.eval()
    oa, aa, ka, _, _ = test(model, device, segments, ground_truth, data)
    print(f"Inference Results: OA: {oa:.4f}, AA: {aa:.4f}, KA: {ka:.4f}")

    # Get the model's predictions (cmap)
    cmap = get_cmap(model, device, segments, data)
    visualize_cmap(cmap, ground_truth, os.path.join(args.output_path, "mgn_visualize_cmap.png"))

    # Save additional outputs
    print(f"Results saved in {args.output_path}")

if __name__ == "__main__":
    inference()
