from .construct_feature_graph import construct_feature_graph
from .segmentation import segmentation
from .find_pca import find_pca
from .results import map_results
from skimage.segmentation import mark_boundaries
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
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
    out=None,
):
    false_image = spy.get_rgb(dataset, [30, 20, 10])

    segments_scales, segments_data, segments_cluster = [], [], []
    
    dataset_pca = find_pca(dataset, 0.999)  # Find PCA

    for idx in tqdm(range(no_clusters)):
        seg = segmentation(
            dataset, ground_truth, segmentation_size * (2 ** (idx + 1)), False
        )

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
        class_map = np.zeros_like(seg, dtype="uint8")
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
        filepath = (
            os.path.join(out, "felz_segmentation_multiscale.png")
            if out
            else "felz_segmentation_multiscale.png"
        )
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
        filepath = (
            os.path.join(out, "felz_label_prop_multiscale.png")
            if out
            else "felz_label_prop_multiscale.png"
        )
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

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
        filepath = (
            os.path.join(out, "felz_boundaries_multiscale.png")
            if out
            else "felz_boundaries_multiscale.png"
        )
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    return segments_scales, segments_data, segments_cluster, segments_results


def optim_scales_felzenswalb(segments_cluster, segments_results, threshold=0.8):
    return [a for (a, b) in zip(segments_cluster, segments_results) if b[0] >= threshold]


def find_valleys(scores):
    scores_array = np.array(scores)
    valleys = np.zeros_like(scores_array, dtype=bool)

    valleys[1:-1] = (scores_array[1:-1] < scores_array[:-2]) & (
        scores_array[1:-1] < scores_array[2:]
    )

    return np.nonzero(valleys)[0]


def peak_indices(scores, threshold=0):
    scores_array = np.array(scores)

    if len(scores_array) < 2:
        return np.array([])

    # Identify valleys
    valley_indices = find_valleys(scores)

    peaks = np.zeros_like(scores_array, dtype=bool)

    # Check internal peaks
    peaks[1:-1] = (scores_array[1:-1] > scores_array[:-2]) & (
        scores_array[1:-1] > scores_array[2:]
    )

    # Check if the first element is a peak
    if scores_array[0] > scores_array[1] + threshold:
        peaks[0] = True

    # Check if the last element is a peak
    if scores_array[-1] > scores_array[-2] + threshold:
        peaks[-1] = True

    peak_indices = np.nonzero(peaks)[0]

    valid_peaks = []
    diff_scores = []

    for peak in peak_indices:
        if peak == 0:  # First peak
            if scores_array[0] - scores_array[1] > threshold:
                valid_peaks.append(peak)
        elif peak == len(scores_array) - 1:  # Last peak
            if scores_array[-1] - scores_array[-2] > threshold:
                valid_peaks.append(peak)
        else:
            # Check both previous and next valleys
            prev_valley_candidates = valley_indices[valley_indices < peak]
            next_valley_candidates = valley_indices[valley_indices > peak]

            if prev_valley_candidates.size > 0 and next_valley_candidates.size > 0:
                prev_valley = scores_array[prev_valley_candidates[-1]]
                next_valley = scores_array[next_valley_candidates[0]]

                diff = (scores_array[peak] - prev_valley) + (
                    scores_array[peak] - next_valley
                )
                if diff > threshold:
                    valid_peaks.append(peak)
                    diff_scores.append(diff)

    return np.array(valid_peaks), np.array(diff_scores), scores_array


def find_optimal_scales(X, max_clusters, threshold=0):
    calinski_harabasz_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        kmeans.fit(X)
        calinski_harabasz_scores.append(calinski_harabasz_score(X, kmeans.labels_))

    return peak_indices(calinski_harabasz_scores, threshold)


def optim_scale_kmeans(
    dataset,
    ground_truth,
    segmentation_size,
    train_size,
    seed,
    beta,
    sigma_s,
    knn_k,
    k,
    no_clusters=5,
    threshold=20,
    verbal=False,
    out=None,
):

    segments = segmentation(dataset, ground_truth, segmentation_size, False)
    unique_labels = np.unique(segments)
    false_image = spy.get_rgb(dataset, [30, 20, 10])

    dataset_pca = find_pca(dataset, 0.999)

    data = construct_feature_graph(
        segments,
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

    max_clusters = min(128, len(np.unique(segments)) // 2)

    X = data.x.cpu()

    peaks, scores, calinski_harabasz_scores = find_optimal_scales(
        X, max_clusters, threshold
    )
    max_indices = np.array([x for _, x in sorted(zip(scores, peaks), reverse=True)])

    segments_scales, segments_data, segments_cluster, segments_results = [], [], [], []

    optimal_clusters = sorted((max_indices + 2)[:no_clusters], reverse=True)

    segments_results = sorted(scores, reverse=True)[:no_clusters]

    for k in optimal_clusters:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        cluster_labels = kmeans.fit_predict(X)

        clustered_img = np.zeros_like(segments, dtype=np.uint8)
        for label in unique_labels:
            clustered_img[segments == label] = cluster_labels[label]

        segments_scales.append(clustered_img)
        segments_cluster.append(k)

        data = construct_feature_graph(
            clustered_img,
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

    class_maps = []

    for data, seg in zip(segments_data, segments_scales):
        class_map = np.zeros_like(seg)
        labels = data.y.cpu() + 1

        for label in np.unique(seg):
            class_map[seg == label] = labels[label]

        class_maps.append(class_map)

    if verbal:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            max_indices,
            np.array(calinski_harabasz_scores)[max_indices],
            color="red",
            s=100,
            zorder=5,
            label="Optimal Scale",
        )
        plt.plot(
            calinski_harabasz_scores,
            marker="o",
            linestyle="-",
            label="Calinski-Harabasz Scores",
        )
        plt.title("Calinski-Harabasz Scores with Maximum Indices Highlighted")
        plt.xlabel("Index")
        plt.ylabel("Calinski-Harabasz Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filepath = (
            os.path.join(out, "kmeans_calinski_harabasz.png")
            if out
            else "kmeans_calinski_harabasz.png"
        )
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

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
        filepath = (
            os.path.join(out, "kmeans_segmentation_multiscale.png")
            if out
            else "kmeans_segmentation_multiscale.png"
        )
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
        filepath = (
            os.path.join(out, "kmeans_label_prop_multiscale.png")
            if out
            else "kmeans_label_prop_multiscale.png"
        )
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

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
        filepath = (
            os.path.join(out, "kmeans_boundaries_multiscale.png")
            if out
            else "kmeans_boundaries_multiscale.png"
        )
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    return segments_scales, segments_data, segments_cluster, segments_results
