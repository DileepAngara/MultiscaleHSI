from utils.load_hsi import load_hsi
from utils.segmentation import segmentation
from utils.visualization import (
    dataset_visualization,
    tsne_visualization,
    label_prop_visualization,
    knn_graph_visualization,
)
from utils.construct_feature_graph import construct_feature_graph
from utils.optimal_clusters import multiscale_felzenswalb, optim_scales_felzenswalb
from utils.find_pca import find_pca
from models import MGNN
from graph_loss import GraphLoss
import torch
import os
import argparse
import sys
from training_loop import train, test
import numpy as np
from utils.results import plot_training_results
from tqdm import tqdm
import random

output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out", type=str, default="mgn_train", help="Name output directory."
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
        default=10,
        help="Select number of clusters for optimal kmeans clustering",
    )
    parser.add_argument(
        "--kmeans_threshold",
        type=int,
        default=25,
        help="Select threshold for optimal kmeans clustering",
    )

    args = parser.parse_args()

    assert (
        args.optimal_clusters_felz is False or args.optimal_clusters_kmeans is False
    ), "Can't use both optimal scale strategies at once"

    id = len(os.listdir(output_dir)) + 1
    out = os.path.join(output_dir, "{}_{}".format(args.out, id))
    if not os.path.exists(out):
        os.mkdir(out)
    sys.stdout = open(os.path.join(out, "log.txt"), "w")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

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
        ground_truth,  # Feature Extraction Pipeline
        args.train_size,
        args.seed,
        args.beta,
        args.sigma_s,
        args.knn_k,
        args.k,
        args.verbal,
        out,
    )

    num_clusters = args.num_clusters

    if args.optimal_clusters_felz:
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
            no_clusters = args.felz_num_clusters,
            threshold = args.felz_threshold,
            verbal = args.verbal,
            out=out
        )
        num_clusters = optim_scales_felzenswalb(segments_cluster, segments_results)

    print("Learned Scales:")
    print("==============================")
    print(num_clusters)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # call model, optimizer, loss function
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

    val_img_dir = os.path.join(out, "val_images")
    if not os.path.exists(val_img_dir):
        os.mkdir(val_img_dir)

    loss_history, acc_history = [], []
    for epoch in tqdm(
        range(args.epochs + 1)
    ):  # train, test loop (in: graph of each band: out: loss, acc)
        loss = train(model, device, optimizer, criterion, data)
        acc, _, _, _ = test(
            model,
            device,
            segments,
            ground_truth,
            data,
            verbal=(epoch % 50 == 0),
            out=val_img_dir,
            figure="epoch_{}".format(epoch),
        )

        loss_history.append(loss)
        acc_history.append(acc)
        if epoch % 50 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    print("Test results:")
    print("==============================")
    test(
        model,
        device,
        segments,
        ground_truth,
        data,
        verbal=True,
        out=out,
        figure="output.png",
    )
    plot_training_results(args.epochs, loss_history, acc_history, out=out)
    torch.save(model.state_dict(), os.path.join(out, "mgn_model.pt"))

    if args.verbal:
        label_prop_visualization(model, segments, ground_truth, data, out)
        tsne_visualization(model, data, out)
        knn_graph_visualization(model, data, out)


if __name__ == "__main__":
    main()
