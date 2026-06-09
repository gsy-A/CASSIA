import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder

from Generate import setup_seed
from Preprocess import preprocess_atac, build_spatial_neighbor_pairs
from scNetwork_spatial import scNetwork


setup_seed(1111)


def load_labels(adata, label_column):

    y = np.asarray(adata.obs[label_column])
    if y.dtype == object or isinstance(y[0], str):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    else:
        y = y.astype(int)

    if y.min() > 0:
        y = y - y.min()
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial ATAC clustering")
    parser.add_argument("--filename", required=True, type=str)
    args, _ = parser.parse_known_args()
    filename = args.filename

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--maxiter", default=150, type=int)
    parser.add_argument("--pretrain_epochs", default=30, type=int)
    parser.add_argument("--gamma", default=[1, 2], nargs=2, type=float)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.001, type=float)

    parser.add_argument("--data_path", default=None)
    parser.add_argument("--atac_file", default="atac_data.h5ad")
    parser.add_argument("--label_column", default="Annotation_for_Combined")
    parser.add_argument("--spatial_key", default="spatial")

    parser.add_argument("--min_cells", default=20, type=int)
    parser.add_argument("--min_peaks", default=10, type=int)
    parser.add_argument("--n_top_peaks", default=50000, type=int)
    parser.add_argument("--spatial_neighbors", default=6, type=int)
    parser.add_argument("--spatial_weight", default=0.1, type=float)

    parser.add_argument("--ae_weights", default=None)
    parser.add_argument("--save_dir", default=f"results/{filename}_spatial")
    parser.add_argument("--ae_weight_file", default=f"weights/{filename}_spatial_ATAC_AEweights.pth.tar")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--ratio", default=0.05, type=float)
    parser.add_argument("--generate", default=14000, type=int)
    parser.add_argument("--margin", default=0.1, type=float)

    args = parser.parse_args()

    atac_path = os.path.join(args.data_path, args.atac_file)
    adata = sc.read_h5ad(atac_path)

    y = load_labels(adata, args.label_column)
    adata.obs["label"] = y

    adata = preprocess_atac(
        adata,
        min_cells=args.min_cells,
        min_peaks=args.min_peaks,
        n_top_peaks=args.n_top_peaks,
    )

    spatial_pairs = build_spatial_neighbor_pairs(
        adata,
        spatial_key=args.spatial_key,
        n_neighbors=args.spatial_neighbors,
    )

    if issparse(adata.X):
        X = np.asarray(adata.X.todense(), dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)
    y = adata.obs["label"].values.astype(int)

    model = scNetwork(
        input_dim=X.shape[1],
        z_dim=32,
        encodeLayer=[256, 128],
        decodeLayer=[128, 256],
        sigma=args.sigma,
        gamma1=args.gamma[0],
        gamma2=args.gamma[1],
        device=args.device,
    )
    weight_dir = os.path.dirname(args.ae_weight_file)
    if weight_dir and not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    if args.ae_weights is None:
        model.pretrain_autoencoder(
            X=X,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            ae_weights=args.ae_weight_file,
        )
    else:
        if not os.path.isfile(args.ae_weights):
            raise ValueError(f"Checkpoint file not found: {args.ae_weights}")
        checkpoint = torch.load(args.ae_weights)
        model.load_state_dict(checkpoint["ae_state_dict"], strict=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    os.makedirs("final", exist_ok=True)

    n_clusters = len(np.unique(y))

    y_pred, nmi, ari, acc, ami, homo, embedding = model.fit(
        X=X,
        n_clusters=n_clusters,
        generate=args.generate,
        margin=args.margin,
        ratio=args.ratio,
        init_centroid=None,
        y_pred_init=None,
        y=y,
        batch_size=args.batch_size,
        num_epochs=args.maxiter,
        update_interval=args.update_interval,
        tol=args.tol,
        save_dir=args.save_dir,
        spatial_pairs=spatial_pairs,
        spatial_weight=args.spatial_weight,
    )

    print("Final Evaluation:")
    print(f"  ACC  = {acc:.4f}")
    print(f"  AMI  = {ami:.4f}")
    print(f"  NMI  = {nmi:.4f}")
    print(f"  ARI  = {ari:.4f}")
    print(f"  Homo = {homo:.4f}")

    pd.DataFrame(
        embedding, columns=[f"dim_{i}" for i in range(embedding.shape[1])]
    ).to_csv(f"final/{filename}_spatial_embedding.csv", index=False)

    pd.DataFrame(
        {"predicted_label": y_pred}
    ).to_csv(f"final/{filename}_spatial_pred.csv", index=False)
