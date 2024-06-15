# Multiscale spectral graph neural networks for hyperspectral imaging

Contributors:
* Yang Tuan Anh
* Phuong Dao
* Hy Truong Son (Correspondent / PI)

## Demo Instructions (not part of final)
Create virtual environment
```bash
py -m venv .venv
.venv\Scripts\activate
```

Installing modules
```bash
pip install -r requirements.txt
```

Training (GCN)
```bash
py src/gcn_train.py --out gcn_train_results --verbal
```

Inference (GCN)
```bash
py src/gcn_infer.py --out gcn_infer_results --verbal --weights output/train_results/gcn_model.pt
```

Training (MGN)
```bash
py src/mgn_train.py --out mgn_train_results --verbal --num_clusters 16 1
```

Training (MGN) with optimal scale selection
```bash
py src/mgn_train.py --out mgn_train_results --verbal --optimal_clusters_felz
```

Training (MGN) with optimal scale selection and parameters
```bash
py src/mgn_train.py --out mgn_train_results --verbal --optimal_clusters_felz --felz_num_clusters 10 --felz_threshold 0.8
```

Inference (MGN)
```bash
py src/mgn_infer.py --out mgn_infer_results --verbal --weights output/mgn_train_results/mgn_model.pt --num_clusters 16 1
```

Benchmarking (for 5% training sample, by default)
```bash
py src/benchmark.py
```

Benchmarking (for 10% training sample)
```bash
py src/benchmark.py --train_size 0.1
```