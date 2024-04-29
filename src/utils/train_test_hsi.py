from sklearn.model_selection import train_test_split
import numpy as np

def summarize_label_counts(label_matrix, mask):
    # Apply the mask on the label matrix
    masked_labels = label_matrix[mask]

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(masked_labels, return_counts=True)

    # Create a dictionary of class counts
    class_count_dict = dict(zip(unique_classes, class_counts))

    return class_count_dict

def train_test_hsi(ground_truth, train_size, random_state=None):
  # Example train-test mask where non-zero classes are marked as True
  train_test_mask = ground_truth != 0

  # Get indices for train-test split on non-zero classes
  non_zero_indices = np.transpose(np.nonzero(train_test_mask))
  if random_state:
    train_idx, test_idx = train_test_split(non_zero_indices, train_size=train_size, random_state=random_state)
  else:
    train_idx, test_idx = train_test_split(non_zero_indices, train_size=train_size)

  # Create empty masks
  train_mask = np.zeros(ground_truth.shape, dtype=bool)
  test_mask = np.zeros(ground_truth.shape, dtype=bool)

  # Set train indices to True in train mask
  for idx in train_idx:
      train_mask[idx[0], idx[1]] = True

  # Set test indices to True in test mask
  for idx in test_idx:
      test_mask[idx[0], idx[1]] = True

  print(summarize_label_counts(ground_truth, train_mask))
  print(summarize_label_counts(ground_truth, test_mask))

  return train_mask, test_mask