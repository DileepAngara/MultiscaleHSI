import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import pandas as pd
import seaborn as sns

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

    ITER
)
from utils.load_hsi import load_hsi, seed_everything
from utils.segmentation import get_false_color, segmentation
from utils.find_pca import find_pca
from utils.visualization import (
    visualize_dataset, 
    visualize_segmentation, 
    visualize_graph, 
    plot_training_results, 
    visualize_cmap,
    visualize_cmap_compare_ground_truth
)
from utils.construct_feature_graph import construct_feature_graph
from utils.construct_graph import construct_graph
from utils.graph_loss import GraphLoss
from utils.training_loop import train, test, get_cmap
from utils.validation import map_results

import torch
from models import MGNN
from torch_geometric.nn.models import GCN
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
    parser.add_argument("--dataset", type=str, required=True, help="Select INDIAN, SALINAS, PAVIA, KENNEDY, BOTSWANA, TORONTO")
    parser.add_argument("--segmentation_size", type=int, required=True, help="Segmentation Size (e.g. 10, 100, 200)")
    parser.add_argument("--num_clusters", type=str, required=True, help="Comma-separated list of cluster numbers (e.g. '5,10,15')")
    parser.add_argument("--sample_size", type=float, required=True, help="Ratio of available training data, within (0, 1)")
    parser.add_argument("--training", action="store_true", help="Train? (Y or N)")
    args = parser.parse_args()

    DATASET = args.dataset
    SEGMENTATION_SIZE = args.segmentation_size
    TRAIN_SIZE = args.sample_size
    OPTIMAL_CLUSTERS = list(map(int, args.num_clusters.split(',')))

    seed_everything(SEED)

    # Create dataset-specific directories
    DATASET_RESULT_PATH = os.path.join(RESULTS_PATH, DATASET)
    os.makedirs(DATASET_RESULT_PATH, exist_ok=True)

    SAMPLE_RESULT_PATH = os.path.join(DATASET_RESULT_PATH, f"sample_{int(TRAIN_SIZE*100)}")
    if not os.path.exists(SAMPLE_RESULT_PATH):
        os.mkdir(SAMPLE_RESULT_PATH)

    # Setup logging with file output
    log_file = os.path.join(SAMPLE_RESULT_PATH, "log.txt")
    setup_logging(log_file)

    dataset, ground_truth = load_hsi(DATASET, DATA_PATH) # Load image, args
    false_color = get_false_color(dataset)
    dataset_pca = find_pca(dataset, 0.999)

    visualize_dataset(dataset, ground_truth, false_color, os.path.join(SAMPLE_RESULT_PATH, "visualize_dataset.png"))

    NFEAT = dataset_pca.shape[2]

    NOUT = len(np.unique(ground_truth[ground_truth!=0]))

    segments = segmentation(dataset, SEGMENTATION_SIZE) # Segmentation (in: img, out: segmentation label)
    visualize_segmentation(segments, false_color, ground_truth, os.path.join(SAMPLE_RESULT_PATH, "visualize_segmentation.png"))

    edge_index, graph = construct_graph(segments)
    visualize_graph(graph, os.path.join(SAMPLE_RESULT_PATH, "visualize_graph.png"))

    data = construct_feature_graph(segments, dataset_pca, ground_truth, # Feature Extraction Pipeline
                                TRAIN_SIZE, SEED, BETA, SIGMA_S, KNN_K, K)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.imshow(ground_truth, cmap="jet")
    plt.axis("off")
    plt.savefig(os.path.join(SAMPLE_RESULT_PATH, "ground_truth.png"), dpi=600)

    logging.info(f"Benchmarking GCN on {TRAIN_SIZE*100}% sample size")
    SAMPLE_GCN_PATH = os.path.join(SAMPLE_RESULT_PATH, "gcn")

    if not os.path.exists(SAMPLE_GCN_PATH):
        os.mkdir(SAMPLE_GCN_PATH)

    gcn_results, gcn_seg_map_list = [], []

    if args.training:
        for idx in tqdm(range(ITER)):
            torch.manual_seed(idx)
            model = GCN(in_channels = NFEAT,
                    hidden_channels = NHID,
                    out_channels = NOUT,
                    num_layers = 2,
                    norm="layernorm").to(device)

            optimizer = torch.optim.Adam(model.parameters())
            criterion = GraphLoss()

            for layer in model.children(): # reset weights
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            for epoch in range(EPOCH+1): # train, test loop (in: graph of each band: out: loss, acc)
                loss = train(model, device, optimizer, criterion, data)

            seg_map = get_cmap(model, device, segments, data)
            gcn_seg_map_list.append(seg_map)

        logging.info(f'Saving to {os.path.join(SAMPLE_GCN_PATH, "gcn_seg_map_list.npy")}')
        np.save(os.path.join(SAMPLE_GCN_PATH, "gcn_seg_map_list.npy"), gcn_seg_map_list)
    else:
        logging.info(f'Loading from {os.path.join(SAMPLE_GCN_PATH, "gcn_seg_map_list.npy")}')
        gcn_seg_map_list = np.load(os.path.join(SAMPLE_GCN_PATH, "gcn_seg_map_list.npy"))

    for seg_map in gcn_seg_map_list:
        oa, aa, ka, report, matrix = map_results(seg_map, ground_truth)
        gcn_results.append([oa, aa, ka, report, matrix])


    gcn_metrics_results = np.array([[oa, aa, ka] for oa, aa, ka, _, _ in gcn_results])
    means = np.mean(gcn_metrics_results.astype(np.double).T, axis=1)
    std_devs = np.std(gcn_metrics_results.astype(np.double).T, axis=1)

    # Print results
    logging.info(f"Means: {means*100}")
    logging.info(f"Standard Deviations: {std_devs*100}")

    plt.imshow(gcn_seg_map_list[-1], cmap="jet", vmin=0)
    plt.axis("off")
    plt.savefig(os.path.join(SAMPLE_GCN_PATH, "gcn_seg_map.png"), dpi=600)

    visualize_cmap_compare_ground_truth(gcn_seg_map_list[-1], ground_truth,
                                    os.path.join(SAMPLE_GCN_PATH, "gcn_compare_ground_truth.png"))
    
    logging.info(f"Benchmarking MGN on {TRAIN_SIZE*100}% sample size")
    SAMPLE_MGN_PATH = os.path.join(SAMPLE_RESULT_PATH, "mgn")

    if not os.path.exists(SAMPLE_MGN_PATH):
        os.mkdir(SAMPLE_MGN_PATH)

    mgn_results, mgn_seg_map_list = [], []

    if args.training:
        for idx in tqdm(range(ITER)):
            torch.manual_seed(idx)
            model = MGNN(nfeat = NFEAT,
                nhid = NHID,
                nout = NOUT,
                dropout = DROPOUT, num_clusters = OPTIMAL_CLUSTERS).to(device)

            optimizer = torch.optim.Adam(model.parameters())
            criterion = GraphLoss()

            for layer in model.children(): # reset weights
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            for epoch in range(EPOCH+1): # train, test loop (in: graph of each band: out: loss, acc)
                loss = train(model, device, optimizer, criterion, data)

            seg_map = get_cmap(model, device, segments, data)
            mgn_seg_map_list.append(seg_map)

        logging.info(f'Saving to {os.path.join(SAMPLE_MGN_PATH, "mgn_seg_map_list.npy")}')
        np.save(os.path.join(SAMPLE_MGN_PATH, "mgn_seg_map_list.npy"), mgn_seg_map_list)
    else:
        logging.info(f'Loading from {os.path.join(SAMPLE_MGN_PATH, "mgn_seg_map_list.npy")}')
        mgn_seg_map_list = np.load(os.path.join(SAMPLE_MGN_PATH, "mgn_seg_map_list.npy"))

    for seg_map in mgn_seg_map_list:
        oa, aa, ka, report, matrix = map_results(seg_map, ground_truth)
        mgn_results.append([oa, aa, ka, report, matrix])

    mgn_metrics_results = np.array([[oa, aa, ka] for oa, aa, ka, _, _ in mgn_results])
    means = np.mean(mgn_metrics_results.astype(np.double).T, axis=1)
    std_devs = np.std(mgn_metrics_results.astype(np.double).T, axis=1)

    # Print results
    logging.info(f"Means: {means*100}")
    logging.info(f"Standard Deviations: {std_devs*100}")

    plt.imshow(mgn_seg_map_list[-1], cmap="jet", vmin=0)
    plt.axis("off")
    plt.savefig(os.path.join(SAMPLE_MGN_PATH, "mgn_seg_map.png"), dpi=600)

    visualize_cmap_compare_ground_truth(mgn_seg_map_list[-1], ground_truth,
                                    os.path.join(SAMPLE_MGN_PATH, "mgn_compare_ground_truth.png"))
    
    logging.info(f"Benchmarking MGN_OPT (Optimal) on {TRAIN_SIZE*100}% sample size")
    SAMPLE_MGN_OPT_PATH = os.path.join(SAMPLE_RESULT_PATH, "mgn_opt")

    if not os.path.exists(SAMPLE_MGN_OPT_PATH):
        os.mkdir(SAMPLE_MGN_OPT_PATH)

    mgn_opt_results, mgn_opt_seg_map_list = [], []

    if args.training:
        for idx in tqdm(range(ITER)):
            torch.manual_seed(idx)
            model = MGNN(nfeat = NFEAT,
                nhid = NHID,
                nout = NOUT,
                dropout = DROPOUT, num_clusters = OPTIMAL_CLUSTERS).to(device)

            optimizer = torch.optim.Adam(model.parameters())
            criterion = GraphLoss()

            for layer in model.children(): # reset weights
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            for epoch in range(EPOCH+1): # train, test loop (in: graph of each band: out: loss, acc)
                loss = train(model, device, optimizer, criterion, data)

            seg_map = get_cmap(model, device, segments, data)
            mgn_opt_seg_map_list.append(seg_map)

        logging.info(f'Saving to {os.path.join(SAMPLE_MGN_OPT_PATH, "mgn_opt_seg_map_list.npy")}')
        np.save(os.path.join(SAMPLE_MGN_OPT_PATH, "mgn_opt_seg_map_list.npy"), mgn_opt_seg_map_list)
    else:
        logging.info(f'Loading from {os.path.join(SAMPLE_MGN_OPT_PATH, "mgn_opt_seg_map_list.npy")}')
        mgn_opt_seg_map_list = np.load(os.path.join(SAMPLE_MGN_OPT_PATH, "mgn_opt_seg_map_list.npy"))

    for seg_map in mgn_opt_seg_map_list:
        oa, aa, ka, report, matrix = map_results(seg_map, ground_truth)
        mgn_opt_results.append([oa, aa, ka, report, matrix])

    mgn_opt_metrics_results = np.array([[oa, aa, ka] for oa, aa, ka, _, _ in mgn_opt_results])
    means = np.mean(mgn_opt_metrics_results.astype(np.double).T, axis=1)
    std_devs = np.std(mgn_opt_metrics_results.astype(np.double).T, axis=1)

    # Print results
    logging.info(f"Means: {means*100}")
    logging.info(f"Standard Deviations: {std_devs*100}")

    plt.imshow(mgn_opt_seg_map_list[-1], cmap="jet", vmin=0)
    plt.axis("off")
    plt.savefig(os.path.join(SAMPLE_MGN_OPT_PATH, "mgn_opt_seg_map.png"), dpi=600)

    visualize_cmap_compare_ground_truth(mgn_opt_seg_map_list[-1], ground_truth,
                                    os.path.join(SAMPLE_MGN_OPT_PATH, "mgn_opt_compare_ground_truth.png"))
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].imshow(ground_truth, cmap="jet", vmin=0)
    ax[1].imshow(gcn_seg_map_list[-1], cmap="jet", vmin=0)
    ax[2].imshow(mgn_seg_map_list[-1], cmap="jet", vmin=0)
    ax[3].imshow(mgn_opt_seg_map_list[-1], cmap="jet", vmin=0)

    for i in range(4):
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLE_RESULT_PATH, "comparison.png"), dpi=600)
    
    benchmark_data = {
        "GCN": gcn_results,
        "MOB-GCN": mgn_results,
        "MOB-GCN (Optimal)": mgn_opt_results
    }

    metrics = ["Overall Accuracy", "Average Accuracy", "Kappa Coefficient"]

    # Calculate mean and std for each model
    model_names = list(benchmark_data.keys())
    means = []
    std_devs = []

    for model in model_names:
        # Convert to numpy array and transpose
        metrics_results = np.array([[oa, aa, ka] for oa, aa, ka, _, _ in benchmark_data[model]])
        means.append(np.mean(metrics_results.astype(np.double).T, axis=1))
        std_devs.append(np.std(metrics_results.astype(np.double).T, axis=1))

    # Convert lists to numpy arrays for easier handling
    means = np.array(means)
    std_devs = np.array(std_devs)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the positions and width for the bars
    bar_width = 0.08
    indices = np.arange(len(metrics))

    # Plot each model's data with error bars
    for i, model in enumerate(model_names):
        ax.bar(indices + i * bar_width, means[i], bar_width, yerr=std_devs[i], label=model, capsize=5)

    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title(f'Model Comparison of 10 Runs - {int(TRAIN_SIZE*100)}% Sample on GCN, MOB-GCN, and MOB-GCN (Optimal)')
    ax.set_xticks(indices + bar_width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLE_RESULT_PATH, "metric_comparison.png"), dpi=600)

    results_table = pd.DataFrame(means, columns=metrics, index=model_names)
    results_table_std = pd.DataFrame(std_devs, columns=[f"{metric} Std" for metric in metrics], index=model_names)

    # Concatenate means and std deviations
    full_results_table = pd.concat([results_table, results_table_std], axis=1)
    full_results_table.to_csv(os.path.join(SAMPLE_RESULT_PATH, "results_table.csv"))

    confusion_matrices = {}
    for model_name, results in benchmark_data.items():
        confusion_matrix_list = []
        for result in results:
            confusion_matrix_list.append(result[4])
        average_confusion_matrix = np.mean(confusion_matrix_list, axis=0)
        confusion_matrices[model_name] = average_confusion_matrix


    for model_name, cm in confusion_matrices.items():
        plt.figure(figsize=(12, 10))  # Increase figure size

        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size": 10})  # Adjust font size for annotations
        plt.title(f'Confusion Matrix for {model_name} of 10 Runs - {int(TRAIN_SIZE*100)}% Sample', fontsize=16)  # Larger title font
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)

        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
        plt.yticks(rotation=0)  # Keep y-axis labels horizontal

        plt.tight_layout()  # Ensure proper layout
        plt.savefig(os.path.join(SAMPLE_RESULT_PATH, f"{model_name}_confusion_matrix.png"), dpi=600)

if __name__ == "__main__":
    main()
