import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
import os
from sklearn.manifold import TSNE
from utils.results import map_results
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import pandas as pd
from collections import defaultdict


def dataset_visualization(dataset, ground_truth, out=None):
    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(spy.get_rgb(dataset, [30, 20, 10]))
    ax[0].set_title("False Color (RGB)")
    ax[0].axis("off")

    ax[1].imshow(np.mean(dataset, axis=2), cmap="jet")
    ax[1].set_title("False Color (Mean)")
    ax[1].axis("off")

    ax[2].imshow(ground_truth, cmap="jet")
    ax[2].set_title("Ground Truth")
    ax[2].axis("off")

    plt.tight_layout()
    filepath = (
        os.path.join(out, "dataset_visualization.png")
        if out
        else "dataset_visualization.png"
    )
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    masked_labels = ground_truth[ground_truth != 0]

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(masked_labels, return_counts=True)

    # Create a dictionary of class counts
    class_count_dict = dict(zip(unique_classes, class_counts))

    print("Class Counts:")
    print("==============================")
    print(class_count_dict)


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
        ax[1].scatter(
            X_tsne[indices, 0],
            X_tsne[indices, 1],
            c=[color],
            label=f"Label {label}",
            s=10,
        )
    ax[1].set_title("TSNE Visualization of Embeddings After Training")
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")

    pred_labels = data.y.cpu() + 1
    for label, color in zip(unique_labels, colors):
        indices = np.where(np.array(pred_labels) == label)
        ax[0].scatter(
            X_tsne[indices, 0],
            X_tsne[indices, 1],
            c=[color],
            label=f"Label {label}",
            s=10,
        )

    ax[0].set_title("TSNE Visualization of Embeddings After Label Prop")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")

    plt.tight_layout()
    filepath = (
        os.path.join(out, "tsne_visualization.png") if out else "tsne_visualization.png"
    )
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


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
    filepath = (
        os.path.join(out, "label_prop_visualization.png")
        if out
        else "label_prop_visualization.png"
    )
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def knn_graph_visualization(model, data, out):
    # Visualize the graph
    X = data.x.cpu()
    model.eval()
    with torch.no_grad():
        logits = model(data)
        labels = logits.argmax(dim=1).cpu() + 1

    # TSNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    G = to_networkx(data, to_undirected=True)
    pos = {node: coords for node, coords in zip(G.nodes(), X_tsne)}
    node_colors = labels

    fig, ax = plt.subplots()

    # Draw the graph with node colors based on labels
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, cmap=plt.cm.jet, node_size=20, ax=ax
    )
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Create a legend
    unique_labels = np.unique(labels)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=plt.cm.jet(label / max(unique_labels)),
            linestyle="",
            markersize=10,
            label=f"class {label}",
        )
        for label in unique_labels
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), title="Classes"
    )

    ax.set_title("Graph with Node Colors Based on y Value")
    plt.axis("on")  # Turns on the axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    filepath = (
        os.path.join(out, "knn_graph_visualization.png")
        if out
        else "knn_graph_visualization.png"
    )
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def performance_visualization(data, out=None):
    metrics = ["Overall Accuracy", "Average Accuracy", "Kappa Coefficient"]

    # Calculate mean and std for each model
    model_names = list(data.keys())
    means = []
    std_devs = []

    for model in model_names:
        # Convert to numpy array and transpose
        model_data = np.array(data[model])[:, :3].astype(np.double).T
        # Calculate mean and std for each row
        means.append(np.mean(model_data, axis=1))
        std_devs.append(np.std(model_data, axis=1))

    # Convert lists to numpy arrays for easier handling
    means = np.array(means)
    std_devs = np.array(std_devs)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the positions and width for the bars
    bar_width = 0.08
    indices = np.arange(len(metrics))

    # Plot each model's data with error bars
    for i, model in enumerate(model_names):
        ax.bar(
            indices + i * bar_width,
            means[i],
            bar_width,
            yerr=std_devs[i],
            label=model,
            capsize=5,
        )

    # Add labels and title
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Scores")
    ax.set_title("Model Comparison")
    ax.set_xticks(indices + bar_width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    filepath = (
        os.path.join(out, "performance_visualization.png")
        if out
        else "performance_visualization.png"
    )
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    results_table = pd.DataFrame(means, columns=metrics, index=model_names)
    results_table_std = pd.DataFrame(
        std_devs, columns=[f"{metric} Std" for metric in metrics], index=model_names
    )

    # Concatenate means and std deviations
    full_results_table = pd.concat([results_table, results_table_std], axis=1)
    filepath = os.path.join(out, "full_results.csv") if out else "full_results.csv"

    print(full_results_table)
    full_results_table.to_csv(filepath, index_label=False)


def class_visualization(data, out=None):
    classification_reports = defaultdict(list)
    for model_name, results in data.items():
        for result in results:
            classification_reports[model_name].append(result[3])

    # Initialize a dictionary to store F1-scores for each class and model
    f1_dict = {}

    # Loop through each model's classification reports
    for model_name, reports in classification_reports.items():
        for report in reports:
            lines = report.split("\n")
            # Extract class-wise F1-score
            for line in lines[2:-5]:
                values = line.split()
                class_label = int(values[0])
                f1_score = float(values[3])
                # Accumulate F1-score for each class across all models
                f1_dict.setdefault((model_name, class_label), []).append(f1_score)

    # Calculate the average F1-score for each class across all models for each model
    average_f1 = {}
    for key in f1_dict.keys():
        f1_avg = np.mean(f1_dict[key])
        average_f1[key] = f1_avg

    # Get unique model names and sort them
    models = sorted(set([key[0] for key in f1_dict.keys()]))

    # Sort classes by label
    sorted_classes = sorted(set([key[1] for key in f1_dict.keys()]))

    # Plot each model's line
    plt.figure(figsize=(10, 6))
    for model in models:
        x = []
        y = []
        for label in sorted_classes:
            if (model, label) in average_f1:
                x.append(label)
                y.append(average_f1[(model, label)])
        plt.plot(x, y, marker="o", label=model)

    plt.title("Average F1-score of Every Class Across Models")
    plt.xlabel("Class Label")
    plt.ylabel("Average F1-score")
    plt.xticks(sorted_classes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filepath = (
        os.path.join(out, "class_visualization.png")
        if out
        else "class_visualization.png"
    )
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    # Create a table of results
    results_table = pd.DataFrame(index=sorted_classes, columns=models)

    # Populate the results table with average F1-scores
    for model in models:
        for label in sorted_classes:
            results_table.at[label, model] = average_f1.get((model, label), np.nan)

    # Print the results table
    results_table

    filepath = (
        os.path.join(out, "full_results_classes.csv")
        if out
        else "full_results_classes.csv"
    )

    print(results_table)
    results_table.to_csv(filepath, index_label=False)
