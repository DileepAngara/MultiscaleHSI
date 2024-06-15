from utils.load_hsi import load_hsi
from utils.segmentation import segmentation
from utils.visualization import (
    dataset_visualization,
    performance_visualization,
    class_visualization,
)
from utils.optimal_clusters import (
    multiscale_felzenswalb,
    optim_scales_felzenswalb,
    optim_scale_kmeans,
)
from utils.construct_feature_graph import construct_feature_graph
from utils.find_pca import find_pca
from models import GCN, MGNN
from graph_loss import GraphLoss
import torch
import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
from utils.results import map_results
from training_loop import train, test
import random

output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out", type=str, default="benchmark", help="Name output directory."
    )
    parser.add_argument("--verbal", action="store_true", help="Select verbal output")

    parser.add_argument(
        "--dataset",
        type=str,
        default="INDIAN",
        help="Select INDIAN, PAVIA or SALINAS for benchmarking.",
    )
    parser.add_argument(
        "--segmentation_size",
        type=int,
        default=10,
        help="Select Felzenswalb minimum segmentation size. (10 for INDIAN, 50 for SALINAS, 100 for PAVIA)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Select random seed for reproducibility"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.05,
        help="Select train size ratio (float from 0 to 1)",
    )

    parser.add_argument(
        "--sigma_s",
        type=float,
        default=0.2,
        help="Select normalization constant sigma_s",
    )
    parser.add_argument(
        "--knn_k", type=int, default=8, help="Select number of neighbors for KNN graph"
    )
    parser.add_argument(
        "--k", type=float, default=15, help="Select normalization constant k"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="Select mean-weighted feature beta ratio",
    )

    parser.add_argument(
        "--nhid", type=int, default=256, help="Select hidden layer size"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Select dropout ratio"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Select number of epochs"
    )

    parser.add_argument(
        "--num_clusters",
        type=int,
        nargs="*",
        default=[10, 5],
        help="Select learned resolutions (default is [10, 5])",
    )
    parser.add_argument(
        "--optimal_clusters_felz",
        action="store_true",
        help="Select optimal clusters with felzenswalb segmentation",
    )
    parser.add_argument(
        "--felz_num_clusters",
        type=int,
        default=10,
        help="Select number of clusters for optimal felz segmentation",
    )
    parser.add_argument(
        "--felz_threshold",
        type=float,
        default=0.8,
        help="Select threshold for optimal felz segmentation",
    )
    parser.add_argument(
        "--optimal_clusters_kmeans",
        action="store_true",
        help="Select optimal clusters with kmeans clustering",
    )
    parser.add_argument(
        "--kmeans_num_clusters",
        type=int,
        default=5,
        help="Select number of clusters for optimal kmeans clustering",
    )
    parser.add_argument(
        "--kmeans_threshold",
        type=int,
        default=20,
        help="Select threshold for optimal kmeans clustering",
    )

    parser.add_argument(
        "--iters", type=int, default=10, help="Select number of iterations"
    )

    args = parser.parse_args()

    id = len(os.listdir(output_dir)) + 1
    out = os.path.join(
        output_dir, "{}_{}_{}".format(int(args.train_size * 100), args.out, id)
    )
    if not os.path.exists(out):
        os.mkdir(out)
    sys.stdout = open(os.path.join(out, "log.txt"), "w")

    dataset, ground_truth = load_hsi(dataset=args.dataset)

    segments = segmentation(
        dataset, ground_truth, args.segmentation_size, verbal=args.verbal, out=out
    )

    if args.verbal:
        dataset_visualization(dataset, ground_truth, out=out)

    dataset_pca = find_pca(dataset, 0.999)  # Find PCA

    data = construct_feature_graph(
        segments,
        dataset_pca,
        ground_truth,
        args.train_size,
        args.seed,
        args.beta,
        args.sigma_s,
        args.knn_k,
        args.k,
        args.verbal,
        out,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_results(data, desc):
        data = np.array(data)
        np.save(os.path.join(out, desc.replace(" ", "_").lower() + ".npy"), data)
        means = np.mean(data[:, :3].astype(np.double).T, axis=1)
        std_devs = np.std(data[:, :3].astype(np.double).T, axis=1)

        print(desc)
        print("==============================")
        print("Means:", means * 100)
        print("Standard Deviations:", std_devs * 100)

    def label_prop_benchmark():
        data = construct_feature_graph(
            segments,
            dataset_pca,
            ground_truth,  # Feature Extraction Pipeline
            args.train_size,
            args.seed,
            args.beta,
            args.sigma_s,
            args.knn_k,
            args.k,
            verbal=False,
            out=out,
        )

        class_map = np.zeros_like(segments, dtype="uint8")
        labels = data.y.cpu() + 1

        for label in np.unique(segments):
            class_map[segments == label] = labels[label]

        return map_results(class_map, ground_truth, False)

    def gcn_benchmark():
        model = GCN(
            nfeat=dataset_pca.shape[2],
            nhid=args.nhid,
            nout=len(np.unique(ground_truth[ground_truth != 0])),
            n_nodes=len(np.unique(segments)),
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = GraphLoss()

        for layer in model.children():  # reset weights
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        for epoch in range(
            args.epochs + 1
        ):  # train, test loop (in: graph of each band: out: loss, acc)
            loss = train(model, device, optimizer, criterion, data)

        return test(model, device, segments, ground_truth, data, verbal=False)

    def mgn_benchmark(num_clusters):
        model = MGNN(
            nfeat=dataset_pca.shape[2],
            nhid=args.nhid,
            nout=len(np.unique(ground_truth[ground_truth != 0])),
            n_nodes=len(np.unique(segments)),
            dropout=args.dropout,
            num_clusters=num_clusters,
            use_norm=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = GraphLoss()

        for layer in model.children():  # reset weights
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        for epoch in range(
            args.epochs + 1
        ):  # train, test loop (in: graph of each band: out: loss, acc)
            loss = train(model, device, optimizer, criterion, data)

        return test(model, device, segments, ground_truth, data, verbal=False)

    def benchmark(func, desc, name, num_clusters = None):
        results = []
        for idx in tqdm(range(args.iters), desc=desc):
            torch.manual_seed(idx)
            random.seed(idx)
            np.random.seed(idx)
            if num_clusters is None:
                oa, aa, ka, report = func()
            else:
                oa, aa, ka, report = func(num_clusters)
            results.append([oa, aa, ka, report])

        print_results(results, name)
        if num_clusters: print("Clusters:", num_clusters)
        return results

    label_prop_results = benchmark(
        label_prop_benchmark, "Benchmarking Label Propagation", "Label Prop results"
    )
    gcn_results = benchmark(gcn_benchmark, "Benchmarking GCN", "GCN results")

    no_classes = len(np.unique(ground_truth[ground_truth != 0]))
    num_classes = [no_classes, 1]

    (_, _, segments_cluster, segments_results) = multiscale_felzenswalb(
        dataset,
        ground_truth,
        args.segmentation_size,
        args.train_size,
        args.seed,
        args.beta,
        args.sigma_s,
        args.knn_k,
        args.k,
        no_clusters=args.felz_num_clusters,
        threshold=args.felz_threshold,
        verbal=args.verbal,
        out=out,
    )
    num_clusters_felz = optim_scales_felzenswalb(segments_cluster, segments_results)

    (_, _, segments_cluster, _) = optim_scale_kmeans(
        dataset,
        ground_truth,
        args.segmentation_size,
        args.train_size,
        args.seed,
        args.beta,
        args.sigma_s,
        args.knn_k,
        args.k,
        no_clusters=args.kmeans_num_clusters,
        threshold=args.kmeans_threshold,
        verbal=args.verbal,
        out=out,
    )
    num_clusters_kmeans = segments_cluster

    mgn_results = benchmark(
        mgn_benchmark, "Benchmarking MGN", "MGN results", num_classes
    )
    mgn_felz_results = benchmark(
        mgn_benchmark,
        "Benchmarking MGN (Felzenswalb Strategy)",
        "MGN Felz results",
        num_clusters_felz
    )
    mgn_kmeans_results = benchmark(
        mgn_benchmark,
        "Benchmarking MGN (Kmeans Strategy)",
        "MGN Kmeans results",
        num_clusters_kmeans
    )

    data = {
        "Label Propagation": label_prop_results,
        "GCN": gcn_results,
        "MGN": mgn_results,
        "MGN (Felzenswalb Strategy)": mgn_felz_results,
        "MGN (Kmeans Strategy)": mgn_kmeans_results,
    }

    print("Model Comparison")
    print("==============================")
    performance_visualization(data, out)

    print("Class Accuracy Across Models Comparison")
    print("==============================")
    class_visualization(data, out)


if __name__ == "__main__":
    main()
