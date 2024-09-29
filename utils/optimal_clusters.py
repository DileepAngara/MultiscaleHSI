import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

def calculate_superpixel_std(hsi_image, superpixel_segments, random_state=None):
    unique_superpixels = np.unique(superpixel_segments)
    superpixel_stds = []

    for sp in unique_superpixels:
        sp_mask = (superpixel_segments == sp)
        sp_pixels = hsi_image[sp_mask]

        # Calculate the standard deviation for the current superpixel
        sp_std = np.std(sp_pixels, axis=0)
        sp_mean = np.mean(sp_pixels, axis=0)
        sp_mean[sp_mean == 0] = 1e-8
        normalized_std = sp_std / sp_mean
        superpixel_stds.append(normalized_std)

    superpixel_stds = np.array(superpixel_stds)

    # Use Isolation Forest to detect and ignore extreme values
    iso_forest = IsolationForest(contamination=0.1, random_state=random_state)  # Adjust the contamination parameter as needed
    outliers = iso_forest.fit_predict(superpixel_stds)

    # Filter out the outliers
    inliers = superpixel_stds[outliers == 1]

    # Calculate the average standard deviation of the inliers
    avg_superpixel_std = np.mean(inliers)

    return avg_superpixel_std


def find_peak_indices(values):
    values = np.array(values)

    # Find peaks: values[i] > values[i-1] and values[i] > values[i+1]
    peaks = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1

    # Sort peaks by their values in descending order
    sorted_peaks = peaks[np.argsort(values[peaks])[::-1]]

    return sorted_peaks


def find_optimal_scale(data, dataset, segments, num_clusters=5, random_state=None):
  superpixel_std = []
  max_clusters = min(128, len(np.unique(segments)) // 2)
  X = data.x.cpu()

  for k in tqdm(range(2, max_clusters + 1)):
      kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
      kmeans.fit(X)
      class_map = np.zeros_like(segments)
      for label in np.unique(segments):
        class_map[segments == label] = kmeans.labels_[label]
      superpixel_std.append(calculate_superpixel_std(dataset, class_map, random_state=random_state))

  CV = np.array(superpixel_std)
  relative_changes = (CV[1:] - CV[:-1]) / CV[:-1]

  peak_indices = find_peak_indices(relative_changes)

  return sorted(peak_indices[:num_clusters] + 3, reverse=True)