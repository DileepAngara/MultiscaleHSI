# Multiscale spectral graph neural networks for hyperspectral imaging

Contributors:
* Yang Tuan Anh
* Phuong Dao
* Hy Truong Son (Correspondent / PI)

## Demo Instructions (not part of final)
Create virtual environment
```
py -m venv .venv
.venv\Scripts\activate
```

Installing modules
```
pip install -r requirements.txt
```

Training (GCN)
```
py src/gcn_train.py --out gcn_train_results --verbal
```

Inference (GCN)
```
py src/gcn_infer.py --out gcn_infer_results --verbal --weights output/train_results/gcn_model.pt
```

Training (MGN)
```
py src/mgn_train.py --out mgn_train_results --verbal --num_clusters 16 1
```

Inference (MGN)
```
py src/mgn_infer.py --out mgn_infer_results --verbal --weights output/mgn_train_results/mgn_model.pt --num_clusters 16 1
```

Benchmarking (for 5% training sample, by default)
```
py src/benchmark.py
```

Benchmarking (for 10% training sample)
```
py src/benchmark.py --train_size 0.1
```