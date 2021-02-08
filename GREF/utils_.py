import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split    
import torch.sparse as ts
import torch.nn.functional as F
import warnings
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor, get_train_val_test, encode_onehot
# from deeprobust.graph import utils
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
from sklearn.linear_model import LogisticRegression

def covert_to_tensor(adjs, features, labels=None, device='cpu'):
    """Convert adjs, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : tuple or list with scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    list
        torch tensor of adjs, features and labels
    """

    if sp.issparse(adjs[0]):
        adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
    else:
        adjs = [torch.FloatTensor(adj) for adj in adjs]

    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return [adj.to(device) for adj in adjs], features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return [adj.to(device) for adj in adjs], features.to(device), labels.to(device)


def load_preattacked_data(dataset, attack, ptb_rate, path="./data/"):
    """load the pre-attacked data from deeprobust

    Parameters
    ----------
    dataset : str
        dataset name.
    attack : str
        attack method name.
    ptb_rate : float
        perturbation rate.
    path : str, optional
        root directory where the dataset should be saved, by default "./data/"

    Returns
    -------
    tuple
        return adj, perturbed_adj, features, labels
    """
    data = Dataset(root=path, name=dataset, setting='nettack', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    if dataset == 'pubmed':
        # just for matching the results in the paper; seed details in https://github.com/ChandlerBang/Pro-GNN/issues/2
        idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],
                                                          val_size=0.1, test_size=0.8, stratify=encode_onehot(labels), seed=15)
        # if attack == 'nettack':
        #     print('use nettack idx_test')
        #     idx_test = perturbed_data.idx['attacked_test_nodes']

    if attack == 'random' and ptb_rate > 0:
        from deeprobust.graph.global_attack import Random
        attacker = Random()
        n_perturbations = int(ptb_rate * (adj.sum()//2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj.tocsr()
        # print(perturbed_adj)

    if attack == 'meta' or attack == 'nettack':

        if ptb_rate > 0:
            perturbed_data = PrePtbDataset(root=path,
                                        name=dataset,
                                        attack_method=attack,
                                        ptb_rate=ptb_rate)
            perturbed_adj = perturbed_data.adj
        # perturbed_adj = adj
        if attack == 'nettack':
            if ptb_rate == 0:
                perturbed_data = PrePtbDataset(root=path,
                                        name=dataset,
                                        attack_method=attack,
                                        ptb_rate=1.0)
            idx_test = perturbed_data.target_nodes
            if idx_test is None:
                print("please update the code by https://github.com/DSE-MSU/DeepRobust/issues/44")

    if attack == 'no' or ptb_rate == 0:
        perturbed_adj = adj
    return (adj, perturbed_adj, features, labels), (idx_train, idx_val, idx_test)
