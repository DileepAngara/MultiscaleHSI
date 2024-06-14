from utils.results import map_results
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def label_prop_visualization(model, segments, ground_truth, data, out=None):
    class_map = np.zeros_like(segments)
    labels = data.y.cpu() + 1

    for label in np.unique(segments):
        class_map[segments == label] = labels[label]

    fig, ax = plt.subplots(1, 5, figsize=(10, 6))

    ax[0].imshow(ground_truth, cmap="jet")
    ax[0].set_title("Ground Truth", size=8)
    ax[0].axis("off")

    ax[1].imshow(class_map, cmap="jet", vmin=0)
    ax[1].set_title("Label Propagation", size=8)
    ax[1].axis("off")

    print("Label Prop Performance before Training")
    print("==============================")
    map_results(class_map, ground_truth, True)

    class_map[ground_truth == 0] = 0
    ax[2].imshow(class_map, cmap="jet")
    ax[2].set_title("Label Propagation (masked)", size=8)
    ax[2].axis("off")

    model.eval()
    with torch.no_grad():
        logits = model(data)
        gnn_labels = logits.argmax(dim=1).cpu() + 1

    gnn_class_map = np.zeros_like(segments)

    for label in np.unique(segments):
        gnn_class_map[segments == label] = gnn_labels[label]

    ax[3].imshow(gnn_class_map, cmap="jet", vmin=0)
    ax[3].set_title("Trained output", size=8)
    ax[3].axis("off")

    print("Trained Performance")
    print("==============================")
    map_results(gnn_class_map, ground_truth, True)

    gnn_class_map[ground_truth == 0] = 0
    ax[4].imshow(gnn_class_map, cmap="jet")
    ax[4].set_title("Trained output (masked)", size=8)
    ax[4].axis("off")

    plt.tight_layout()
    filepath = os.path.join(out, "label_prop_visualization.png") if out else "label_prop_visualization.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()