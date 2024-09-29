import argparse
import numpy as np
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from utils.load_hsi import load_hsi, seed_everything
from utils.segmentation import segmentation
from utils.find_pca import find_pca
from utils.construct_feature_graph import construct_feature_graph
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
from utils.optimal_clusters import calculate_superpixel_std, find_peak_indices

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

def find_optimal_scale(data, dataset, segments, num_clusters=5, random_state=None):
    superpixel_std = []
    max_clusters = min(128, len(np.unique(segments)) // 2)
    X = data.x.cpu()

    for k in tqdm(range(2, max_clusters + 1)):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
        kmeans.fit(X)
        class_map = np.zeros_like(segments)
        for label in np.unique(segments):
            class_map[segments == label] = kmeans.labels_[label]
        superpixel_std.append(calculate_superpixel_std(dataset, class_map, random_state=random_state))

    CV = np.array(superpixel_std)
    relative_changes = (CV[1:] - CV[:-1]) / CV[:-1]

    peak_indices = find_peak_indices(relative_changes)

    return sorted(peak_indices[:num_clusters] + 3, reverse=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Select dataset: INDIAN, SALINAS, PAVIA, KENNEDY, BOTSWANA, TORONTO")
    parser.add_argument("--segmentation_size", type=int, required=True, help="Segmentation Size (e.g. 10, 100, 200)")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters to consider for optimal scale")
    parser.add_argument("--random_state", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    # Set random seed
    seed_everything(SEED)

    # Load dataset and perform PCA
    dataset, ground_truth = load_hsi(args.dataset, DATA_PATH)
    dataset_pca = find_pca(dataset, 0.999)
    # Perform segmentation
    segments = segmentation(dataset, args.segmentation_size)

    # Construct feature graph
    data = construct_feature_graph(segments, dataset_pca, ground_truth, TRAIN_SIZE, SEED, BETA, SIGMA_S, KNN_K, K)
    
    # Find optimal scale
    optimal_scale = find_optimal_scale(data, dataset, segments, num_clusters=args.num_clusters, random_state=args.random_state)

    optimal_scale = [int(scale) for scale in optimal_scale]
    
    print(f"Optimal Scale: {optimal_scale}")

if __name__ == "__main__":
    main()
