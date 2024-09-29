import numpy as np
import random
import os

import torch
from PIL import Image

from scipy.io import loadmat
import rasterio
import wget


def load_hsi(dataset, dataset_path):
  if dataset not in ["INDIAN", "PAVIA", "SALINAS", "BOTSWANA", "KENNEDY", "TORONTO"]:
    assert "Please select either INDIAN, PAVIA, SALINAS, BOTSWANA, KENNEDY or TORONTO for benchmarking"

  if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

  def download_if_not_exist(filename, file_url):
    if not os.path.exists(filename):
        wget.download(file_url, out=dataset_path)

  match dataset:
    case "INDIAN":
      corrected_file = os.path.join(dataset_path, 'Indian_pines_corrected.mat')
      corrected_url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"

      gt_file = os.path.join(dataset_path, 'Indian_pines_gt.mat')
      gt_url = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

      download_if_not_exist(corrected_file, corrected_url)
      download_if_not_exist(gt_file, gt_url)

      dataset = loadmat(corrected_file)['indian_pines_corrected']
      ground_truth = loadmat(gt_file)['indian_pines_gt']
    case "PAVIA":
      corrected_file = os.path.join(dataset_path, 'Pavia.mat')
      corrected_url = "https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat"

      gt_file = os.path.join(dataset_path, 'Pavia_gt.mat')
      gt_url = "https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat"

      download_if_not_exist(corrected_file, corrected_url)
      download_if_not_exist(gt_file, gt_url)

      dataset = loadmat(corrected_file)['pavia']
      ground_truth = loadmat(gt_file)['pavia_gt']
    case "SALINAS":
      corrected_file = os.path.join(dataset_path, 'Salinas_corrected.mat')
      corrected_url = "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"

      gt_file = os.path.join(dataset_path, 'Salinas_gt.mat')
      gt_url = "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"

      download_if_not_exist(corrected_file, corrected_url)
      download_if_not_exist(gt_file, gt_url)

      dataset = loadmat(corrected_file)['salinas_corrected']
      ground_truth = loadmat(gt_file)['salinas_gt']
    case "KENNEDY":
      corrected_file = os.path.join(dataset_path, 'KSC.mat')
      corrected_url = "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat"

      gt_file = os.path.join(dataset_path, 'KSC_gt.mat')
      gt_url = "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat"

      download_if_not_exist(corrected_file, corrected_url)
      download_if_not_exist(gt_file, gt_url)

      dataset = loadmat(corrected_file)['KSC']
      ground_truth = loadmat(gt_file)['KSC_gt']
    case "BOTSWANA":
      corrected_file = os.path.join(dataset_path, 'Botswana.mat')
      corrected_url = "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat"

      gt_file = os.path.join(dataset_path, 'Botswana_gt.mat')
      gt_url = "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat"

      download_if_not_exist(corrected_file, corrected_url)
      download_if_not_exist(gt_file, gt_url)

      dataset = loadmat(corrected_file)['Botswana']
      ground_truth = loadmat(gt_file)['Botswana_gt']
    case "TORONTO":
      gt_file = os.path.join(dataset_path, "Toronto_reference.tif")
      ground_truth = Image.open(gt_file)
      ground_truth = np.array(ground_truth)
      values = np.unique(ground_truth)

      ground_truth[ground_truth==255] = 0
      for idx, val in enumerate(values[:-1], start=1):
        ground_truth[ground_truth==val] = idx
      
      corrected_file = os.path.join(dataset_path, 'suburban/suburban/20170820_Urban_Ref_Reg_Subset.tif')
      with rasterio.open(corrected_file) as src:
        dataset = src.read()
      dataset = np.array(dataset)

      dataset = np.transpose(dataset, (1, 2, 0))
      ground_truth = np.pad(ground_truth, ((0, 34), (0, 1)), mode='constant', constant_values=0)
    case _:
      dataset, ground_truth = None, None
      pass

  return dataset, ground_truth

def seed_everything(seed):
  """
  Seeds basic parameters for reproducibility of results
  """
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False