# Hyperparameter tuning using bio-inspired algorithms in GCN and GAN models for link prediction in a PPI network 

## Overview

This project is made to compare different biologically inspired algorithms like Genetic Algorithm, Particle Swarm Optimization, Ant Colony Optimization in hyperparameter tuning in Graph Convolutional Network and Generative Adversarial Network models for link prediction in a SNAPS protein-protein interaction network. 

## 🧬 Protein-Protein Interaction Network: PP-Pathways
This repository explores the <a href="https://snap.stanford.edu/biodata/datasets/10000/10000-PP-Pathways.html">PP-Pathways dataset from the Stanford SNAP BioData collection</a>. It represents a large-scale protein-protein interaction (PPI) network derived from pathway databases.

<ul>
<li>Nodes: 21,554 proteins</li>
<li>Edges: 342,338 interactions</li>
<li>Data Type: Undirected, unweighted graph</li>
<li>Source: Pathway-based protein associations</li>
<li>Format: Edge list (.csv) with each row representing a protein-protein interaction</li>
</ul>

### Graph visualization for the PPI network
<img src="ppi-visualization.png" alt="Protein Graph" width="500"/>
This image was done using <a href="https://cytoscape.org/">Cytoscape</a>.

## Results
These tables showcase the best hyperparameter results chosen with each algorithm and comparing them using metrics such as F-1 score, AUC-ROC score and Loss. Each algortihm is done on 10 epochs in 30 evaluations.
<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr style="background-color: #d3d3d3;">
      <th rowspan="2">Model</th>
      <th rowspan="2">Metric</th>
      <th>None</th>
      <th>GA</th>
      <th>PSO</th>
      <th>ABC</th>
      <th>Simulated Annealing</th>
      <th>Hill Climbing</th>
      <th>Random Search</th>
      <th>ACO</th>
      <th>Bayesian Search</th>
      <th>Optuna</th>
    </tr>
  </thead>
  <tbody>
    <!-- GCN Performance -->
    <tr><td rowspan="7" style="background-color:#d9f2e6;">GCN</td><td>f1</td><td>0.8024</td><td>0.8479</td><td>0.8498</td><td>0.849</td><td>0.8393</td><td>0.8491</td><td>0.8411</td><td>0.8526</td><td>0.8524</td><td>0.8519</td></tr>
    <tr><td>auc</td><td>0.8739</td><td>0.9151</td><td>0.8287</td><td>0.895</td><td>0.8354</td><td>0.7861</td><td>0.905</td><td>0.9175</td><td>0.9179</td><td>0.9144</td></tr>
    <tr><td>loss</td><td>1.2882</td><td>1.2605</td><td>1.3822</td><td>31.2253</td><td>1.381</td><td>1.376</td><td>1.239</td><td>11.2008</td><td>1.2434</td><td>1.2382</td></tr>
    <tr><td>hidden channels</td><td>256</td><td>143</td><td>32</td><td>112</td><td>58</td><td>125</td><td>64</td><td>144</td><td>144</td><td>45</td></tr>
    <tr><td>learning rate</td><td>0.01</td><td>0.00448</td><td>0.01579</td><td>0.02367</td><td>0.031</td><td>0.02508</td><td>0.0172</td><td>0.01</td><td>0.00264</td><td>0.00698</td></tr>
    <tr><td>layers / dropout</td><td>5 / 0</td><td>3 / 0.35</td><td>4 / 0.3</td><td>3 / 0.48</td><td>3 / 0.55</td><td>3 / 0.54</td><td>3 / 0.5</td><td>4 / 0.3</td><td>3 / 0.05</td><td>4 / 0.29</td></tr>
    <tr><td>time</td><td>58s</td><td>9m 28s</td><td>10m 56s</td><td>7m 13s</td><td>9m 25s</td><td>8m 53s</td><td>5m 10s</td><td>5m 23s</td><td>4m 30s</td><td>8m 23s</td></tr>
  </tbody>

<tbody>
  <tr><td rowspan="7" style="background-color:#e0e6f2;">GAN</td><td>f1</td><td>0.7277</td><td>0.8476</td><td>0.7497</td><td>0.748</td><td>0.6804</td><td>0.7494</td><td>0.7568</td><td>0.7743</td><td>0.7458</td><td>0.7499</td></tr>
    <tr><td>auc</td><td>0.7345</td><td>0.5</td><td>0.554</td><td>0.5756</td><td>0.5</td><td>0.5</td><td>0.7725</td><td>0.7728</td><td>0.7669</td><td>0.7727</td></tr>
    <tr><td>average loss</td><td>0.0466</td><td>31.2233</td><td>0.0264</td><td>0.0275</td><td>-3.8527</td><td>-4.2903</td><td>0.0348</td><td>0.1738</td><td>0.1828</td><td>0.1771</td></tr>
    <tr><td>hidden channels</td><td>256</td><td>202</td><td>277</td><td>405</td><td>242</td><td>398</td><td>416</td><td>512</td><td>160</td><td>373</td></tr>
    <tr><td>learning rate</td><td>1e-4</td><td>0.00629</td><td>0.003</td><td>0.00276</td><td>1e-5</td><td>0.00158</td><td>0.0006</td><td>0.0002</td><td>0.00028</td><td>0.0006</td></tr>
    <tr><td>dropout</td><td>0.3</td><td>0.6</td><td>0.45</td><td>0.47</td><td>0.1</td><td>0.16</td><td>0.32</td><td>0.2</td><td>0.35</td><td>0.28</td></tr>
    <tr><td>time</td><td>1m 21s</td><td>14m 49s</td><td>10m 6s</td><td>11m 11s</td><td>9m 18s</td><td>11m 47s</td><td>10m 23s</td><td>16m 59s</td><td>5m 40s</td><td>16m 24s</td></tr>
</tbody>
</table>



## Quickstart

### Prerequisites
- Python 3.8+
- [scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/get-started/locally/) 
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- [NiaPy](https://niapy.org/en/stable/index.html#niapy-s-documentation)
- [scikit-optimize](https://scikit-optimize.github.io/stable/)

### Installation
```bash
git clone https://github.com/milagjurovska/PPI-link-prediction-with-optimized-gcn-and-gan.git
cd PPI-link-prediction-with-optimized-gcn-and-gan
