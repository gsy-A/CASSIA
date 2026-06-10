# CASSIA

CASSIA (Chromatin Accessibility Semi-Supervised Identity-Aware representation learning), a label-guided regulatory representation learning framework for single-cell and spatial ATAC-seq data. For single-cell ATAC-seq, CASSIA uses limited high-confidence biological anchors to guide the learning of accessibility representations that better separate regulatory cell states while preserving data-driven discovery. For spatial ATAC-seq, CASSIA further incorporates spatial neighborhood information to encourage locally coherent regulatory domains in tissue space.

# Contents

- [Framework diagram](#framework_diagram)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Output](#output)
- [Citation](#citation)
- 
# Framework diagram
<img width="4917" height="2616" alt="CASSIA" src="https://github.com/user-attachments/assets/86cc75e0-a479-4995-8361-92229a674ce2" />

Dependencies

Python 3.9.23

Pytorch 2.8.0

Pytorch Geometric 2.6.1

Scanpy 1.10.3

Sklearn 1.6.1

Numpy 2.0.2

Pandas 2.3.3

All experiments of CASSIA in this study are conducted on Nvidia 5090 GPU. We suggest to install the dependencies in a conda environment (conda create -n CASSIA).

Usage

Run CASSIA on a single-cell ATAC-seq dataset
python main.py
Run CASSIA on a spatial ATAC-seq dataset
python main_spatial.py 

After running the command, the learned embeddings, predicted cell identities, and evaluation results will be saved in the specified output directory.
