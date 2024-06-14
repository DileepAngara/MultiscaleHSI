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

Training
```
py src/train.py --out train_results --verbal
```

Inference
```
py src/infer.py --out infer_results --verbal --weights output/train_results/gcn_model.pt
```

Benchmarking (for 5% training sample, by default)
```
py src/benchmark.py
```

Benchmarking (for 10% training sample)
```
py src/benchmark.py --train_size 0.1
```