from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import watershed
from skimage.restoration import estimate_sigma
from skimage.filters import sobel
import numpy as np
import matplotlib.pyplot as plt

def segmentation_map(segments, ground_truth, clip=False, label_mask=None):
  unique_labels = np.unique(segments)
  if np.min(unique_labels) != 0:
    segments -= np.min(unique_labels)
    unique_labels -= np.min(unique_labels)

  class_map = []
  for label in unique_labels:
    pos = np.bincount(ground_truth[segments == label]).argmax()
    if label_mask:
      class_map.append(pos * label_mask[label])
    else:
      class_map.append(pos)

  class_map = np.array([[class_map[col] for col in row] for row in segments])

  if clip:
    class_map[ground_truth==0] = 0

  print("Number of nodes:", len(unique_labels))
  return class_map

def slic_seg(img, size):
  X = img.copy()

  noise = np.array(estimate_sigma(X, channel_axis=2))

  reshaped_weights = noise.reshape(1, 1, img.shape[2])

  X_gradient = np.sum(X / reshaped_weights, axis=2)

  lbp = local_binary_pattern(X_gradient, 8*3, 3)

  segments_slic_lbp = slic(lbp, n_segments=size, compactness=1e-5, slic_zero=True, start_label=0)

  plt.imshow(mark_boundaries(X_gradient, segments_slic_lbp))

  return segments_slic_lbp

def felzenswalb_seg(img, size):
  segments_fz = felzenszwalb(img, sigma=0.95, min_size=size)

  return segments_fz

def watershed_seg(img, size):
  X = img.copy()

  noise = np.array(estimate_sigma(X, channel_axis=2))

  reshaped_weights = noise.reshape(1, 1, img.shape[2])

  X_gradient = np.sum(X / reshaped_weights, axis=2)

  gradient = sobel(X_gradient)

  segments_watershed = watershed(gradient, compactness=1e-5, markers=size)

  return segments_watershed

def segmentation(img, method="FELZENSWALB", size=20):
  normalized_img = np.rint((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(int)
  false_image = plt.get_cmap('viridis')(normalized_img[:,:,0])[:, :, :3]
  seg = None
  match method:
    case "FELZENSWALB":
      seg = felzenswalb_seg(img, size)
    case "SLIC":
      seg = slic_seg(img, size)
    case "WATERSHED":
      seg = watershed_seg(img, size)
    case _:
      assert "Pick between FELZENSWALB (default if unselected), SLIC, WATERSHED"
  plt.imshow(mark_boundaries(false_image, seg))
  plt.show()

  map = segmentation_map(seg, ground_truth)
  plt.imshow(map, cmap="jet")
  plt.colorbar()
  plt.show()

  results(map, ground_truth)
  return seg