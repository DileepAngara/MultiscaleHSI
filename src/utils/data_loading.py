from scipy.io import loadmat

def load_hsi(path="data", dataset="INDIAN"):
  if dataset != "INDIAN" or dataset != "PAVIAN" or dataset != "SALINAS":
    assert "Please select either INDIAN, PAVIAN or SALINAS for benchmarking"

  match dataset:
    case "INDIAN":
      dataset = loadmat(f'{path}/Indian_pines_corrected.mat')['indian_pines_corrected']
      ground_truth = loadmat(f'{path}/Indian_pines_gt.mat')['indian_pines_gt']
      print(f'Dataset: {dataset.shape}\nGround Truth: {ground_truth.shape}')
    case "PAVIAN":
      dataset = loadmat(f'{path}Pavia.mat')['pavia']
      ground_truth = loadmat(f'{path}Pavia_gt.mat')['pavia_gt']
      print(f'Dataset: {dataset.shape}\nGround Truth: {ground_truth.shape}')
    case "SALINAS":
      dataset = loadmat(f'{path}Salinas_corrected.mat')['salinas_corrected']
      ground_truth = loadmat(f'{path}Salinas_gt.mat')['salinas_gt']
      print(f'Dataset: {dataset.shape}\nGround Truth: {ground_truth.shape}')
    case _:
      pass

  return dataset, ground_truth