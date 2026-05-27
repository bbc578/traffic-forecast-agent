# Raw Real Datasets

Place externally downloaded real traffic datasets here. This repository does not include METR-LA,
PEMS-BAY, or other large real datasets.

Expected examples:

```text
data/raw/METR-LA/metr-la.h5
data/raw/METR-LA/adj_mx.pkl
data/raw/PEMS-BAY/pems-bay.h5
data/raw/PEMS-BAY/adj_mx.pkl
```

Convert them with `python -m traffic_agent.data.prepare_real_dataset`. If the files are missing,
training on real configs must fail with a clear error rather than silently falling back to synthetic data.
