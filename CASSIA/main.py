import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import os
import torch

from scNetwork import scNetwork
from Generate import *
from Preprocess import preprocess_atac


setup_seed(1111)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ATAC clustering')
    
    parser.add_argument('--filename', default=None, type=str)
    args, _ = parser.parse_known_args()
    filename = args.filename
    
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=150, type=int)
    parser.add_argument('--pretrain_epochs', default=30, type=int)
    parser.add_argument('--gamma', default=[1, 2], nargs=2, type=float) 
    parser.add_argument('--sigma', default=0.5, type=float)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--atac_file', default='atac_data.h5ad')
    parser.add_argument('--meta_file', default='meta_data.csv') 
    parser.add_argument('--label_column', default='Cluster')
    
    parser.add_argument('--min_cells', default=20, type=int)
    parser.add_argument('--min_peaks', default=10, type=int)
    
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default=f'results/{filename}')
    parser.add_argument('--ae_weight_file', default=f'weights/{filename}_ATAC_AEweights.pth.tar')  
    
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--ratio', default=0.05, type=float)
    parser.add_argument('--generate', default=14000, type=int) 
    parser.add_argument('--margin', default=0.1, type=float)
    
    args = parser.parse_args()
    
    atac_path = os.path.join(args.data_path, args.atac_file)
    adata = sc.read_h5ad(atac_path)
    
    meta_path = os.path.join(args.data_path, args.meta_file)
    meta_data = pd.read_csv(meta_path)
    y_raw = np.array(meta_data[args.label_column])

    y_names = np.array([str(label) for label in y_raw])
    unique_label_names = sorted(pd.unique(y_names))
    label_to_numeric = {label: idx for idx, label in enumerate(unique_label_names)}
    y = np.array([label_to_numeric[label] for label in y_names]).astype(int)
    
    adata.obs['label'] = y
    
    adata = preprocess_atac(
        adata,
        min_cells=args.min_cells,
        min_peaks=args.min_peaks,
        n_top_peaks=50000
    )
    
    from scipy.sparse import issparse
    if issparse(adata.X):
        X = np.array(adata.X.todense()).astype(np.float32)
    else:
        X = np.array(adata.X).astype(np.float32)
    
    y = adata.obs['label'].values.astype(int)

    model = scNetwork(
        input_dim=X.shape[1],
        z_dim=32,
        encodeLayer=[256, 128],
        decodeLayer=[128, 256],
        sigma=args.sigma,
        gamma1=args.gamma[0], 
        gamma2=args.gamma[1], 
        device=args.device
    )

    weight_dir = os.path.dirname(args.ae_weight_file)
    print(weight_dir)
    if weight_dir and not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    if args.ae_weights is None:
        model.pretrain_autoencoder(
            X=X,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            ae_weights=args.ae_weight_file
        )
    else:
        if os.path.isfile(args.ae_weights):
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'], strict=False)
        else:
            raise ValueError(f"Checkpoint file not found: {args.ae_weights}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
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
        save_dir=args.save_dir
    )

    print('Final Evaluation:')
    print(f'  ACC  = {acc:.4f}')
    print(f'  AMI  = {ami:.4f}')
    print(f'  NMI  = {nmi:.4f}')
    print(f'  ARI  = {ari:.4f}')
    print(f'  Homo = {homo:.4f}')

    os.makedirs('final', exist_ok=True)

    embedding_df = pd.DataFrame(
    embedding,
    columns=[f'dim_{i}' for i in range(embedding.shape[1])]
    )
    embedding_path = os.path.join('./final/', f'{filename}_embedding.csv')
    embedding_df.to_csv(embedding_path, index=False)

    pred_df = pd.DataFrame({
        'predicted_label': y_pred
    })
    pred_path = os.path.join('./final/', f'{filename}_pred.csv')
    pred_df.to_csv(pred_path, index=False)


