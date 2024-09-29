import torch
import numpy as np
from utils.validation import map_results


def train(model, device, optim, criterion, data):
  model.train()

  data = data.to(device)

  optim.zero_grad()
  out = model(data.x, data.edge_index, data.edge_attr)
  loss = criterion(out, data.y, data)
  loss.backward()
  optim.step()

  return loss.item()


def test(model, device, segments, ground_truth, data):
  model.eval()

  with torch.no_grad():
    data = data.to(device)
    logits = model(data.x, data.edge_index, data.edge_attr)
    pred = logits.argmax(dim=1).cpu()

  cmap = np.zeros_like(segments)
  unique_labels = np.unique(segments)
  for label in unique_labels:
    cmap[segments == label] = pred[label].item()

  cmap += data.one_index

  # Select predictions and ground truth corresponding to test mask
  cmap_test = cmap[data.test_mask]
  ground_truth_test = ground_truth[data.test_mask]

  return map_results(cmap_test, ground_truth_test)


def get_cmap(model, device, segments, data):
  model.eval()

  with torch.no_grad():
    data = data.to(device)
    logits = model(data.x, data.edge_index, data.edge_attr)
    pred = logits.argmax(dim=1).cpu()

  cmap = np.zeros_like(segments)
  unique_labels = np.unique(segments)
  for label in unique_labels:
    cmap[segments == label] = pred[label].item()

  cmap += data.one_index

  return cmap