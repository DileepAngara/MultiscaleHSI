from scipy.io import loadmat
import os
import wget


def load_hsi(dataset="INDIAN", path=None):
    DATA_PATH = os.path.join(path, "data") if path else "data"

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    if dataset != "INDIAN" or dataset != "PAVIA" or dataset != "SALINAS":
        assert "Please select either INDIAN, PAVIA or SALINAS for benchmarking"

    def download_if_not_exist(filename, file_url):
        if not os.path.exists(filename):
            corrected_filename = wget.download(file_url, out=DATA_PATH)

    match dataset:
        case "INDIAN":
            corrected_file = os.path.join(DATA_PATH, "Indian_pines_corrected.mat")
            corrected_url = (
                "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
            )

            gt_file = os.path.join(DATA_PATH, "Indian_pines_gt.mat")
            gt_url = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

            download_if_not_exist(corrected_file, corrected_url)
            download_if_not_exist(gt_file, gt_url)

            dataset = loadmat(corrected_file)["indian_pines_corrected"]
            ground_truth = loadmat(gt_file)["indian_pines_gt"]
        case "PAVIA":
            corrected_file = os.path.join(DATA_PATH, "Pavia.mat")
            corrected_url = "https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat"

            gt_file = os.path.join(DATA_PATH, "Pavia_gt.mat")
            gt_url = "https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat"

            download_if_not_exist(corrected_file, corrected_url)
            download_if_not_exist(gt_file, gt_url)

            dataset = loadmat(corrected_file)["pavia"]
            ground_truth = loadmat(gt_file)["pavia_gt"]
        case "SALINAS":
            corrected_file = os.path.join(DATA_PATH, "Salinas_corrected.mat")
            corrected_url = (
                "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"
            )

            gt_file = os.path.join(DATA_PATH, "Salinas_gt.mat")
            gt_url = "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"

            download_if_not_exist(corrected_file, corrected_url)
            download_if_not_exist(gt_file, gt_url)

            dataset = loadmat(corrected_file)["salinas_corrected"]
            ground_truth = loadmat(gt_file)["salinas_gt"]
        case _:
            pass

    return dataset, ground_truth
