# Multiscale spectral graph neural networks for hyperspectral imaging

Contributors:
* Yang Tuang Anh
* Hy Truong Son (Correspondent / PI)

## Demo Instructions (not part of final)
Installing modules
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

Benchmarking (for 5% training sample, by default)
```
py src/benchmark.py
```

Benchmarking (for 10% training sample)
```
py src/benchmark.py --train_size 0.1
```