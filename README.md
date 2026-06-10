# CASSIA

CASSIA (Chromatin Accessibility Semi-Supervised Identity-Aware representation learning), a label-guided regulatory representation learning framework for single-cell and spatial ATAC-seq data. For single-cell ATAC-seq, CASSIA uses limited high-confidence biological anchors to guide the learning of accessibility representations that better separate regulatory cell states while preserving data-driven discovery. For spatial ATAC-seq, CASSIA further incorporates spatial neighborhood information to encourage locally coherent regulatory domains in tissue space.

# Contents

- [Framework diagram](#framework_diagram)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Arguments](#Arguments)
- [Output](#output)

## Framework diagram
<img width="4917" height="2616" alt="CASSIA" src="https://github.com/user-attachments/assets/86cc75e0-a479-4995-8361-92229a674ce2" />

## Dependencies

Python 3.9.23

Pytorch 2.8.0

Pytorch Geometric 2.6.1

Scanpy 1.10.3

Sklearn 1.6.1

Numpy 2.0.2

Pandas 2.3.3

All experiments of CASSIA in this study are conducted on Nvidia 5090 GPU. We suggest to install the dependencies in a conda environment (conda create -n CASSIA).

## Usage

Run CASSIA on a single-cell ATAC-seq dataset

python main.py
Run CASSIA on a spatial ATAC-seq dataset

python main_spatial.py

## Arguments

### Common arguments

--filename: dataset name. This argument is used to construct the default data path, result path, and autoencoder weight file path.

--batch_size: batch size for model training. Default: 256.

--maxiter: maximum number of training or clustering iterations. Default: 150.

--pretrain_epochs: number of epochs for autoencoder pre-training. Default: 30.

--gamma: weights of loss terms used in model training. Default: 1 2.

--sigma: sigma parameter used in the model. Default: 0.5.

--update_interval: interval for updating clustering assignments. Default: 1.

--tol: tolerance threshold for convergence. Default: 0.001.

--data_path: path to the input dataset folder.

--atac_file: name of the input ATAC-seq data file. Default: atac_data.h5ad.

--label_column: column name of cell labels or annotations used for evaluation.

--min_cells: minimum number of cells required for peak filtering. Default: 20.

--min_peaks: minimum number of peaks required for cell filtering. Default: 10.

--ae_weights: path to the pre-trained autoencoder weights. Default: None.

--save_dir: directory to store the output results.

--ae_weight_file: path to save or load the autoencoder weight file.

--device: device used for model training. Default: cuda.

--ratio: ratio of labeled cells or identity anchors used in semi-supervised learning. Default: 0.05.

--generate: number of generated or selected training samples. Default: 14000.

--margin: margin parameter used in the triplet-based loss. Default: 0.1.

### Single-cell ATAC-seq arguments

--meta_file: metadata file containing cell annotations. Default: meta_data.csv.

--label_column: column name of cell-type labels in the metadata file. Default: Cluster.

### Spatial ATAC-seq arguments

--spatial_key: key in `adata.obsm` that stores spatial coordinates. Default: spatial.

--spatial_neighbors: number of nearest spatial neighbors used to construct spatial neighbor pairs. Default: 6.

--spatial_weight: weight of the spatial neighborhood consistency regularization term. Default: 0.1.

--label_column: column name of spatial spot annotations. Default: Annotation_for_Combined.


## Output
After running the command, the learned embeddings and predicted cell identities will be saved in the specified output directory.
