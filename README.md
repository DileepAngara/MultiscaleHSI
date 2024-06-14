# Multiscale spectral graph neural networks for hyperspectral imaging

Contributors:
* Yang Tuang Anh
* Hy Truong Son (Correspondent / PI)

## Demo (not part of final)
Install modules
```
pip install -r requirements.txt
```

Training
```
py src/train.py --out train_results --verbal
```

Testing
```
py src/test.py --out test_results --verbal --weights output/train_results/gcn_model.pt
```

Benchmarking
```
py src/benchmark.py
```