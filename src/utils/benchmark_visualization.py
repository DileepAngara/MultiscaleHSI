import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
from collections import defaultdict
import re

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
        ax.bar(indices + i * bar_width, means[i], bar_width, yerr=std_devs[i], label=model, capsize=5)

    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(indices + bar_width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    filepath = os.path.join(out, "performance_visualization.png") if out else "performance_visualization.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

    results_table = pd.DataFrame(means, columns=metrics, index=model_names)
    results_table_std = pd.DataFrame(std_devs, columns=[f"{metric} Std" for metric in metrics], index=model_names)

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
            lines = report.split('\n')
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
        plt.plot(x, y, marker='o', label=model)

    plt.title('Average F1-score of Every Class Across Models')
    plt.xlabel('Class Label')
    plt.ylabel('Average F1-score')
    plt.xticks(sorted_classes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(out, "class_visualization.png") if out else "class_visualization.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

    # Create a table of results
    results_table = pd.DataFrame(index=sorted_classes, columns=models)

    # Populate the results table with average F1-scores
    for model in models:
        for label in sorted_classes:
            results_table.at[label, model] = average_f1.get((model, label), np.nan)

    # Print the results table
    results_table

    filepath = os.path.join(out, "full_results_classes.csv") if out else "full_results_classes.csv"
    
    print(results_table)
    results_table.to_csv(filepath, index_label=False)