# GREF
A PyTorch implementation of "GREF: Graph Data Refinement via Structure and Attribute Reconciliation for Robust Graph Neural Networks".
## Abstact
Recent years have witnessed great success of graph neural networks (GNNs) in various graph data mining tasks. However, studies demonstrate that GNNs are vulnerable to imperceptible structural perturbations. Carefully-crafted perturbations of few edges can significantly degrade the performance of GNNs in downstream tasks. Many useful defense methods have been developed to eliminate the impacts of adversarial edges. However, existing approaches fail to sufficiently exploit the mutual corroboration effects of structures and attributes for graph purification. This paper presents GREF, a novel graph data refinement framework defending GNNs against structural poisoning attacks through structure and attribute reconciliation. We first augment the graph data by two auxiliary graphs, including a structural neighborhood graph and an attributive neighborhood graph. Then, we propose a graph purification scheme which exploits the reconciliation of available structural and attributive data to prune and reweigh the edges. GREF can significantly mitigate the inconsistency between structural and attributive data, which reduces the impacts of adversarial and noisy edges on message-passing in GNNs. Experimental results show that GREF outperforms state-of-the-art defense methods against various adversarial attacks and exhibits significant superior for attacks with high perturbation rates. 
![](./GREF.PNG)
## Requirements and Installation:
To run the code, first you need to install DeepRobust:
```
pip install deeprobust
```
You can see requriements in https://github.com/DSE-MSU/DeepRobust/blob/master/requirements.txt

## Run the code
After installation
```
python run_gref.py --dataset cora --attack meta --ptb_rate 0.25
```
or
```
python run_gref.py --dataset cora --attack nettack --ptb_rate 5.0
```
or
```
python run_gref.py --dataset cora --attack random --ptb_rate 0.2
```