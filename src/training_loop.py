import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.results import map_results
import os


def train(model, device, optim, criterion, data):
    model.train()

    data = data.to(device)

    optim.zero_grad()
    out = model(data)
    loss = criterion(out, data.y, data)
    loss.backward()
    optim.step()

    return loss.item()


def test(
    model, device, segments, ground_truth, data, verbal=False, out=None, figure=None
):
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        logits = model(data)
        pred = logits.argmax(dim=1)

    cmap = np.zeros_like(segments)
    unique_labels = np.unique(segments)
    for label in unique_labels:
        cmap[segments == label] = pred[label]

    cmap += data.one_index

    # Select predictions and ground truth corresponding to test mask
    cmap_test = cmap[data.test_mask]
    ground_truth_test = ground_truth[data.test_mask]

    # Calculate accuracy
    oa, aa, ka, report = map_results(cmap_test, ground_truth_test, verbal)

    if verbal:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cmap, cmap="jet", vmin=0)
        ax[0].set_title("Output")
        ax[0].axis("off")

        cmap_clipped = cmap
        cmap_clipped[ground_truth == 0] = 0
        ax[1].imshow(cmap_clipped, cmap="jet", vmin=0)
        ax[1].set_title("Ground Truth")
        ax[1].axis("off")

        plt.tight_layout()
        filepath = os.path.join(out, figure) if out else figure
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    return oa, aa, ka, report
