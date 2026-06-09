from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.neighbors import NearestNeighbors


def preprocess_atac(adata, min_cells=1, min_peaks=10, n_top_peaks=50000):
    import scanpy as sc
    from scipy.sparse import issparse

    sc.pp.filter_cells(adata, min_counts=min_peaks)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if issparse(adata.X):
        X = adata.X
        mean = np.asarray(X.mean(axis=0)).flatten()
        sq = X.power(2).mean(axis=0)
        var = np.asarray(sq - mean**2).flatten()
    else:
        mean = adata.X.mean(axis=0)
        var = adata.X.var(axis=0)

    cv = var / (mean + 1e-6)
    top_peaks_idx = np.argsort(cv)[-n_top_peaks:]
    adata = adata[:, top_peaks_idx]

    adata.X = (adata.X > 0).astype(np.float32)
    adata.raw = adata.copy()

    return adata


def build_spatial_neighbor_pairs(adata, spatial_key="spatial", n_neighbors=6):
    if spatial_key not in adata.obsm:
        raise ValueError(f"Cannot find spatial coordinates in adata.obsm['{spatial_key}']")

    coords = np.asarray(adata.obsm[spatial_key])

    n_obs = coords.shape[0]
    if n_obs < 2:
        return np.empty((0, 2), dtype=np.int64)

    k = min(max(1, n_neighbors), n_obs - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords)
    indices = nn.kneighbors(return_distance=False)

    pairs = set()
    for i in range(n_obs):
        for j in indices[i, 1:]:
            a, b = sorted((int(i), int(j)))
            if a != b:
                pairs.add((a, b))

    pairs = np.array(sorted(pairs), dtype=np.int64)
    return pairs
