from sklearn.metrics import accuracy_score
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.results import results

def train_tmgnn(model, device, optim, criterion, timeframe_data):
  model.train()

  optim.zero_grad()

  for data in timeframe_data:
    data = data.to(device)
    out = model(data)

  loss = criterion(out, data.y, data)
  loss.backward()
  optim.step()

  return loss.item()

def test_tmgnn(model, device, segments, ground_truth, timeframe_data, report=False):
  model.eval()

  with torch.no_grad():
    for data in timeframe_data:
      data = data.to(device)
      logits = model(data)

    pred = logits.argmax(dim=1)

  cmap = np.array([pred[row].cpu() for row in segments]) + data.one_index

  # Select predictions and ground truth corresponding to test mask
  cmap_test = cmap[data.test_mask]
  ground_truth_test = ground_truth[data.test_mask]

  # Calculate accuracy
  acc = accuracy_score(cmap_test, ground_truth_test)

  if report:
    results(cmap_test, ground_truth_test)

    plt.imshow(cmap, cmap="jet", vmin=0)
    plt.colorbar()
    plt.show()

    cmap_clipped = cmap
    cmap_clipped[ground_truth==0] = 0
    plt.imshow(cmap_clipped, cmap="jet", vmin=0)
    plt.colorbar()
    plt.show()

  return acc