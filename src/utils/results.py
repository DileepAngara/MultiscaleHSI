import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
import spectral as spy
import os
import pandas as pd


def map_results(seg_map, ground_truth, verbal=False):
    seg_map = np.array(seg_map).flatten()
    ground_truth = np.array(ground_truth).flatten()

    non_zero_indices = np.where(ground_truth != 0)
    filtered_ground_truth = ground_truth[non_zero_indices]
    filtered_seg_map = seg_map[non_zero_indices]

    overall_acc = accuracy_score(filtered_ground_truth, filtered_seg_map)
    class_acc = balanced_accuracy_score(filtered_ground_truth, filtered_seg_map)
    kappa_score = cohen_kappa_score(filtered_ground_truth, filtered_seg_map)
    report = classification_report(
        filtered_ground_truth, filtered_seg_map, labels=np.unique(filtered_seg_map)
    )
    if verbal:
        print(report)
        print("OA:", overall_acc)
        print("AA:", class_acc)
        print("KA:", kappa_score)

    return overall_acc, class_acc, kappa_score, report


def plot_training_results(EPOCH, loss_history, accuracy_history, out=None):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss history
    axs[0].plot(range(1, EPOCH + 2), loss_history, "b", label="Training loss")
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot accuracy history
    axs[1].plot(range(1, EPOCH + 2), accuracy_history, "r", label="Training accuracy")
    axs[1].set_title("Training Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    filepath = os.path.join(out, "training_plot.png") if out else "training_plot.png"
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(
        {
            "epoch": np.arange(len(loss_history)) + 1,
            "loss": loss_history,
            "accuracy": accuracy_history,
        }
    )
    filepath = (
        os.path.join(out, "training_record.csv") if out else "training_record.csv"
    )
    df.to_csv(filepath, index_label=False)
