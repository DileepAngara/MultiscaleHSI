from .construct_feature_graph import construct_feature_graph
from .segmentation import segmentation
from .find_pca import find_pca
from .results import map_results
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import spectral as spy
import numpy as np
import math
from tqdm import tqdm
import os

def multiscale_felzenswalb(
    dataset,
    ground_truth,
    segmentation_size,
    train_size,
    seed,
    beta,
    sigma_s,
    knn_k,
    k,
    no_clusters=10,
    threshold=0.8,
    verbal=False,
    out=None
):
    segments_scales, segments_data, segments_cluster = [], [], []

    dataset_pca = find_pca(dataset, 0.999)  # Find PCA

    for idx in tqdm(range(no_clusters)):
        seg = segmentation(dataset, ground_truth, segmentation_size * (2 ** (idx + 1)), False)

        if segments_cluster != []:
            if len(np.unique(seg)) == segments_cluster[-1]:
                continue

        if len(np.unique(seg)) < 2:
            break

        segments_scales.append(seg)
        segments_cluster.append(len(np.unique(seg)))

        data = construct_feature_graph(
            seg,
            dataset_pca,
            ground_truth,
            train_size,
            seed,
            beta,
            sigma_s,
            knn_k,
            k,
            False,
        )

        segments_data.append(data)

    segments_results, class_maps = [], []

    for data, seg in zip(segments_data, segments_scales):
        class_map = np.zeros_like(seg,dtype='uint8')
        labels = data.y.cpu() + 1

        for label in np.unique(seg):
            class_map[seg == label] = labels[label]

        aa, oa, ka, _ = map_results(class_map, ground_truth, False)

        segments_results.append([aa, oa, ka])
        class_maps.append(class_map)

    if verbal:
        no_scales = len(segments_scales)

        for cluster, result in zip(segments_cluster, segments_results):
            print(f"Label Propagation at {cluster}: {result}")

        no_row = math.ceil(no_scales / 4)
        fig, ax = plt.subplots(no_row, 4, figsize=(3 * 4, 3 * no_row))
        ax = ax.flatten()

        for idx, seg in enumerate(segments_scales):
            ax[idx].set_title(f"Segmentation at {segments_cluster[idx]}")
            ax[idx].axis("off")
            ax[idx].imshow(seg, cmap="jet", vmin=0)

        for i in range(idx + 1, no_row * 4):
            ax[i].axis("off")

        plt.tight_layout()
        filepath = os.path.join(out, "segmentation_multiscale.png") if out else "segmentation_multiscale.png"
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        no_row = math.ceil((no_scales + 1) / 4)
        fig, ax = plt.subplots(no_row, 4, figsize=(3 * 4, 3 * no_row))
        ax = ax.flatten()

        ax[0].imshow(ground_truth, cmap="jet", vmin=0)
        ax[0].set_title("Ground Truth", size=8)
        ax[0].axis("off")

        for idx, seg in enumerate(segments_scales):
            ax[idx + 1].imshow(class_maps[idx], cmap="jet", vmin=0)
            ax[idx + 1].set_title(f"Label Propagation at {segments_cluster[idx]}")
            ax[idx + 1].axis("off")

        for i in range(idx + 1, no_row * 4):
            ax[i].axis("off")

        plt.tight_layout()
        filepath = os.path.join(out, "label_prop_multiscale.png") if out else "label_prop_multiscale.png"
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        false_image = spy.get_rgb(dataset, [30, 20, 10])

        no_row = math.ceil((no_scales + 1) / 4)
        fig, ax = plt.subplots(no_row, 4, figsize=(3 * 4, 3 * no_row))
        ax = ax.flatten()

        ax[0].imshow(ground_truth, cmap="jet", vmin=0)
        ax[0].set_title("Ground Truth", size=8)
        ax[0].axis("off")

        for idx, seg in enumerate(segments_scales):
            ax[idx + 1].imshow(mark_boundaries(false_image, seg))
            ax[idx + 1].set_title(f"Segmentation at {segments_cluster[idx]}")
            ax[idx + 1].axis("off")

        for i in range(idx + 1, no_row * 4):
            ax[i].axis("off")

        plt.tight_layout()
        filepath = os.path.join(out, "boundaries_multiscale.png") if out else "boundaries_multiscale.png"
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    return segments_scales, segments_data, segments_cluster, segments_results


def optim_scales_felzenswalb(segments_cluster, segments_results, threshold=0.8):
    return np.array(
        [a for (a, b) in zip(segments_cluster, segments_results) if b[0] >= threshold]
    )
