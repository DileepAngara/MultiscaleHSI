from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
import numpy as np
import matplotlib.pyplot as plt
import spectral as spy
from utils.results import map_results
import os


def segmentation(img, ground_truth, size, verbal=False, out=None):
    segments = felzenszwalb(img, sigma=0.95, min_size=size, channel_axis=2)

    if verbal:
        false_image = spy.get_rgb(img, [30, 20, 10])

        fig, ax = plt.subplots(1, 2)
        boundaries = mark_boundaries(false_image, segments)
        ax[0].imshow(boundaries)
        ax[0].set_title("Segmentation Boundary")
        ax[0].axis("off")

        unique_labels = np.unique(segments)
        seg_map = np.zeros_like(segments)

        for label in unique_labels:
            mask = segments == label
            seg_map[mask] = np.bincount(ground_truth[mask]).argmax()

        ax[1].imshow(seg_map, cmap="jet")
        ax[1].set_title("Segmentation Performance")
        ax[1].axis("off")

        plt.tight_layout()
        filepath = os.path.join(out, "segmentation.png") if out else "segmentation.png"
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        print("Segmentation results:")
        print("==============================")
        map_results(seg_map, ground_truth, verbal)

    return segments
