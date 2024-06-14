from utils.load_hsi import load_hsi
from utils.segmentation import segmentation
from utils.visualize import visualize
from utils.construct_feature_graph import construct_feature_graph
from utils.find_pca import find_pca
from models import GCN
from graph_loss import GraphLoss
import torch
import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
from utils.results import map_results
from training_loop import train, test
from utils.benchmark_visualization import performance_visualization, class_visualization

output_dir = "output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out', type=str, default="benchmark", help='Name output directory.')
    parser.add_argument('--verbal', action='store_true', help='Select verbal output')

    parser.add_argument('--dataset', type=str, default="INDIAN", help='Select INDIAN, PAVIA or SALINAS for benchmarking.')
    parser.add_argument('--segmentation_size', type=int, default=10,
                        help='Select Felzenswalb minimum segmentation size. (10 for INDIAN, 50 for SALINAS, 100 for PAVIA)')
    
    parser.add_argument('--seed', type=int, default=42, help="Select random seed for reproducibility")
    parser.add_argument('--train_size', type=float, default=0.05, help="Select train size ratio (float from 0 to 1)")

    parser.add_argument('--sigma_s', type=float, default=0.2, help="Select normalization constant sigma_s")
    parser.add_argument('--knn_k', type=int, default=8, help="Select number of neighbors for KNN graph")
    parser.add_argument('--k', type=float, default=15, help="Select normalization constant k")
    parser.add_argument('--beta', type=float, default=0.9, help="Select mean-weighted feature beta ratio")

    parser.add_argument('--nhid', type=int, default=256, help="Select hidden layer size")
    parser.add_argument('--dropout', type=float, default=0.5, help="Select dropout ratio")
    parser.add_argument('--epochs', type=int, default=200, help="Select number of epochs")
    parser.add_argument('--iters', type=int, default=10, help="Select number of iterations")

    args = parser.parse_args()

    id = len(os.listdir(output_dir))+1
    out = os.path.join(output_dir, "{}_{}_{}".format(int(args.train_size*100), args.out, id))
    if not os.path.exists(out):
        os.mkdir(out)
    sys.stdout = open(os.path.join(out, "log.txt"), "w")

    dataset, ground_truth = load_hsi(dataset=args.dataset)

    segments = segmentation(dataset, 
                            ground_truth,
                            args.segmentation_size, 
                            verbal=args.verbal, out=out) 

    if args.verbal:
        visualize(dataset, ground_truth, out=out)

    dataset = find_pca(dataset, 0.999) # Find PCA

    data = construct_feature_graph(segments, dataset, ground_truth, args.train_size, args.seed, 
                                   args.beta, args.sigma_s, args.knn_k, args.k, args.verbal, out)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def print_results(data, desc):
        data = np.array(data)
        np.save(os.path.join(out, desc.replace(' ', '_').lower()+".npy"), data)
        means = np.mean(data[:, :3].astype(np.double).T, axis=1)
        std_devs = np.std(data[:, :3].astype(np.double).T, axis=1)

        print(desc)
        print("==============================")
        print("Means:", means*100)
        print("Standard Deviations:", std_devs*100)

    def label_prop_benchmark():
        data = construct_feature_graph(segments, dataset, ground_truth, # Feature Extraction Pipeline
                               args.train_size, args.seed, args.beta, args.sigma_s, args.knn_k, args.k, verbal=False, out=out)

        class_map = np.zeros_like(segments)
        labels = data.y.cpu() + 1

        for label in np.unique(segments):
            class_map[segments == label] = labels[label]

        return map_results(class_map, ground_truth, False)
    
    def gcn_benchmark():
        model = GCN(nfeat = dataset.shape[2],
                nhid = args.nhid,
                nout = len(np.unique(ground_truth[ground_truth!=0])),
                n_nodes = len(np.unique(segments)),
                dropout=args.dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = GraphLoss()

        for layer in model.children(): # reset weights
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for epoch in range(args.epochs+1): # train, test loop (in: graph of each band: out: loss, acc)
            loss = train(model, device, optimizer, criterion, data)

        return test(model, device, segments, ground_truth, data, verbal=False)
    
    def benchmark(func, desc, name):
        results = []
        for idx in tqdm(range(args.iters), desc=desc):
            torch.manual_seed(idx)
            oa, aa, ka, report = func()
            results.append([oa, aa, ka, report])

        print_results(results, name)
        return results
    
    label_prop_results = benchmark(label_prop_benchmark, "Benchmarking Label Propagation:", "Label Prop results")
    gcn_results = benchmark(gcn_benchmark, "Benchmarking GCN:", "GCN results")

    data = {
        "Label Propagation": label_prop_results,
        "GCN": gcn_results,
    }

    print("Model Comparison")
    print("==============================")
    performance_visualization(data, out)

    print("Class Accuracy Across Models Comparison")
    print("==============================")
    class_visualization(data, out)

if __name__ == '__main__':
    main()