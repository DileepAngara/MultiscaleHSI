from sklearn.model_selection import train_test_split
import numpy as np

def train_test_hsi(ground_truth, train_size, random_state=None):
  # Example train-test mask where non-zero classes are marked as True
  train_test_mask = ground_truth != 0

  # Get indices for train-test split on non-zero classes
  non_zero_indices = np.transpose(np.nonzero(train_test_mask))
  train_idx, test_idx = train_test_split(non_zero_indices, train_size=train_size, random_state=random_state)

  # Create empty masks
  train_mask = np.zeros(ground_truth.shape, dtype=bool)
  test_mask = np.zeros(ground_truth.shape, dtype=bool)

  # Set train indices to True in train mask
  for idx in train_idx:
      train_mask[idx[0], idx[1]] = True

  # Set test indices to True in test mask
  for idx in test_idx:
      test_mask[idx[0], idx[1]] = True

  return train_mask, test_mask