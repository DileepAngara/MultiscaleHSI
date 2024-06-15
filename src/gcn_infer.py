from utils.load_hsi import load_hsi
from utils.segmentation import segmentation
from utils.visualization import (
    dataset_visualization,
    tsne_visualization,
    label_prop_visualization,
    knn_graph_visualization,
)
from utils.construct_feature_graph import construct_feature_graph
from utils.find_pca import find_pca
from models import GCN
from graph_loss import GraphLoss
import torch
import os
import argparse
import sys
from training_loop import train, test
import numpy as np
import random

output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out", type=str, default="gcn_infer", help="Name output directory."
    )
    parser.add_argument("--weights", type=str, help="Select Pytorch .pt file")
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

    parser.add_argument("--verbal", action="store_true", help="Select verbal output")
    args = parser.parse_args()

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

    dataset = find_pca(dataset, 0.999)  # Find PCA

    data = construct_feature_graph(
        segments,
        dataset,
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # call model, optimizer, loss function
    model = GCN(
        nfeat=dataset.shape[2],
        nhid=args.nhid,
        nout=len(np.unique(ground_truth[ground_truth != 0])),
        n_nodes=len(np.unique(segments)),
        dropout=args.dropout,
    ).to(device)

    model.load_state_dict(torch.load(args.weights))

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

    model.eval()
    with torch.no_grad():
        logits = model(data)
        gnn_labels = logits.argmax(dim=1).cpu() + 1

    inference_class_map = np.zeros_like(segments)

    for label in np.unique(segments):
        inference_class_map[segments == label] = gnn_labels[label]

    np.save(os.path.join(out, "inference.npy"), inference_class_map)

    if args.verbal:
        label_prop_visualization(model, segments, ground_truth, data, out)
        tsne_visualization(model, data, out)
        knn_graph_visualization(model, data, out)


if __name__ == "__main__":
    main()
