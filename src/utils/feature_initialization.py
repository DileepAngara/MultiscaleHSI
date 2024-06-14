import numpy as np
import torch
from skimage.segmentation import find_boundaries

def calc_weight(vector1, vector2, alpha=1):
    norm_diff = np.linalg.norm(vector1 - vector2)
    return np.exp(-norm_diff ** 2/alpha)

def generate_features(segments, img, alpha=1):
  unique_labels = np.unique(segments)
  mean_features = np.zeros((len(unique_labels), img.shape[2]))

  for label in unique_labels:
    mask = (segments == label)
    masked_image = img * np.expand_dims(mask, axis=-1)
    mean_features[label] = np.mean(masked_image, axis=(0, 1))

  weighted_features = np.zeros((len(unique_labels), img.shape[2]))
  centroids = []

  for label in unique_labels:
    neighbors = find_boundaries(segments == label, connectivity=1)
    neighbor_labels = np.unique(segments[neighbors])

    rows, cols = np.where(segments == label)
    centroids.append((np.mean(rows), np.mean(cols)))

    neighbor_vectors = mean_features[neighbor_labels]
    mean_vector = mean_features[label]

    neighbor_weights = [(calc_weight(mean_vector, neighbor_vector, alpha)) for neighbor_vector in neighbor_vectors]
    neighbor_weights = np.array(neighbor_weights) / np.sum(neighbor_weights)
    if np.sum(neighbor_weights) > 0:
      weighted_features[label] = np.sum(mean_features[neighbor_labels] * neighbor_weights[...,np.newaxis], axis=0)
    else:
      weighted_features[label] = mean_vector # Weights arent strong enough then consider it as its own neighborhood

  return torch.tensor(mean_features, dtype=torch.float), torch.tensor(weighted_features, dtype=torch.float), torch.tensor(np.array(centroids), dtype=torch.float)

def superpixel_classes(segments, ground_truth, train_mask):
    unique_labels = np.unique(segments)
    superpixel_classes = []
    label_mask = []
    non_zero_classes = np.unique(ground_truth[ground_truth != 0])

    for label in unique_labels:
        mask = segments == label
        if np.any(mask & train_mask):
            masked_ground_truth = ground_truth[mask & train_mask]
            if len(masked_ground_truth) > 0:
                class_counts = np.bincount(masked_ground_truth)
                most_common_class = np.argmax(class_counts)
                superpixel_classes.append(most_common_class)
                label_mask.append(True)
            else:
                superpixel_classes.append(np.random.choice(non_zero_classes))
                label_mask.append(False)
        else:
            superpixel_classes.append(np.random.choice(non_zero_classes))
            label_mask.append(False)

    return torch.tensor(superpixel_classes), label_mask

def generate_weights(mean_features, weighted_features, centroids, edge_index, beta = 0.5, sigma_s = 1):
  edge_attr = []
  for i, j in zip(edge_index[0], edge_index[1]):
    mean_vec_norm = np.linalg.norm(mean_features[i]-mean_features[j])
    weighted_vec_norm = np.linalg.norm(weighted_features[i]-weighted_features[j])
    sij = np.exp(((beta-1)* mean_vec_norm**2 - beta * weighted_vec_norm**2)/sigma_s**2)

    centroid_vec_norm = np.linalg.norm(centroids[i]-centroids[j])
    lij = np.exp(-centroid_vec_norm**2)

    edge_attr.append(sij*lij)

  return torch.tensor(np.array(edge_attr), dtype=torch.float)