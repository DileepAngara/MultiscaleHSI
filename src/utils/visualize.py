import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
import os

def visualize(dataset, ground_truth, out=None):
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
  filepath = os.path.join(out, "visualize.png") if out else "visualize.png"
  plt.savefig(filepath, bbox_inches='tight')
  plt.close()

  masked_labels = ground_truth[ground_truth!=0]

  # Get unique classes and their counts
  unique_classes, class_counts = np.unique(masked_labels, return_counts=True)

  # Create a dictionary of class counts
  class_count_dict = dict(zip(unique_classes, class_counts))

  print("Class Counts:")
  print("==============================")
  print(class_count_dict)