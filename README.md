## Introduction
This project aims at introducing graph neural networks and [Deep Graph Library](https://www.dgl.ai/ "Deep Graph Library") to communication systems.  **The content continues updating.**

## Specific Information
1. Folder **Supervised** implemented the MLP and GNN for K-user interference channel power control, trained with supervised learning.

2. Folder **D2D, Cell-free, and Hybrid** is the Pytorch implementation for reproducing the results in  
> Yifei Shen, Jun Zhang, S. H. Song, Khaled B. Letaief (2022).  Graph Neural Networks for Wireless Communications: From Theory to Practice.

- Folder **D2D** implemented MLP, Edge convolution, and proposed GNN for D2D power control.
- Folder **Cell-free** implemented MLP, Heterogenous GNN, and proposed GNN for power control in cell-free massive MIMO.
- Folder **Hybrid** implemented MLP and proposed unrolling method for hybrid precoding.
- The dataset used in the paper can be downloaded at [this link](https://drive.google.com/file/d/1ZcuaRiU0BIyjnUxE7DCaP3B2vRCPD6N7/view?usp=sharing "this link").