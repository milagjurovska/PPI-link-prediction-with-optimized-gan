# Hyperparameter tuning using bio-inspired algorithms in GCN and GAN models for link prediction in a PPI network 

## Overview

This project is made to compare different biologically inspired algorithms like Genetic Algorithm, Particle Swarm Optimization, Ant Colony Optimization in hyperparameter tuning in Graph Convolutional Network and Generative Adversarial Network models for link prediction in a <a href="https://snap.stanford.edu/biodata/datasets/10000/10000-PP-Pathways.html">SNAPS protein-protein interaction network</a>. 


## Quickstart

### Prerequisites
- Python 3.8+
- [PyTorch](https://pytorch.org/get-started/locally/) 
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)

### Installation
```bash
# Clone with datasets 
git clone https://github.com/milagjurovska/PPI-link-prediction-with-optimized-gcn-and-gan.git
cd PPI-link-prediction-with-optimized-gcn-and-gan

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-geometric
