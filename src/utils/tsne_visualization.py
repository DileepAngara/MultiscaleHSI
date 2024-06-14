from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def tsne_visualization(model, data, out):
    X = data.x.cpu()
    model.eval()
    with torch.no_grad():
        logits = model(data)
        labels = logits.argmax(dim=1).cpu() + 1

    # TSNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Get the color values for each label
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    # Create a scatter plot with colored labels
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for label, color in zip(unique_labels, colors):
        indices = np.where(np.array(labels) == label)
        ax[1].scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=[color], label=f'Label {label}', s=10)
    ax[1].set_title("TSNE Visualization of Embeddings After Training")
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")

    pred_labels = data.y.cpu() + 1
    for label, color in zip(unique_labels, colors):
        indices = np.where(np.array(pred_labels) == label)
        ax[0].scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=[color], label=f'Label {label}', s=10)

    ax[0].set_title("TSNE Visualization of Embeddings After Label Prop")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")

    plt.tight_layout()
    filepath = os.path.join(out, "tsne_visualization.png") if out else "tsne_visualization.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()