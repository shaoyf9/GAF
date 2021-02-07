import math
from deeprobust.graph import utils
from . import utils_

from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.stats import entropy
import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

class GraphRefine:
    def __init__(self, adj, features, labels, idx_train, 
                 idx_val=None, idx_test=None, k=15, sth=0.01, tau=1.0):
        self.k, self.sth, self.tau = k, sth, tau
        self.idx_train, self.idx_val, self.idx_test = idx_train, idx_val, idx_test
        self.features = features
        self.labels = labels
        self.adj = adj
        self.sim_matrix = cosine_similarity(features)
        self.adj_s = self.__construct_structure_nerghbor_graph()
        self.adj_a = self.__construct_attribute_nerghbor_graph()
        self.__generate_proba()
    
    def __construct_structure_nerghbor_graph(self):
        adj_2 = self.adj @ self.adj + self.adj
        adj_2[np.diag_indices_from(adj_2)] = 0
        adj_2.eliminate_zeros()
        return adj_2

    def __construct_attribute_nerghbor_graph(self):
        adj_a = sp.eye(self.adj.shape[0])
        adj_a = adj_a.tocsr()
        k = -1 * (self.k + 1)
        index_array = np.argpartition(self.sim_matrix, kth=k, axis=-1)[:, k:]
        sim_values = np.take_along_axis(self.sim_matrix, index_array, axis=-1)
        np.put_along_axis(adj_a, index_array, values=sim_values, axis=-1)
        adj_a.data = np.where(adj_a.data > self.sth, 1.0, 0.0)
        adj_a.setdiag(0.0)
        adj_a.eliminate_zeros()
        return adj_a

    def __generate_proba(self):
        lr = LogisticRegression()
        lr.fit(self.features[self.idx_train], self.labels[self.idx_train])

        # print(lr.score(self.features[self.idx_train], self.labels[self.idx_train]))
        # print(lr.score(features[idx_test], labels[idx_test]))
        self.proba = lr.predict_proba(self.features)
        self.classifer = lr

    def data_refine(self, prune=True, reweight=True):
        if prune:
            # print("start prune")
            self.adj = self.edge_pruning(self.adj)
            self.adj_a = self.edge_pruning(self.adj_a)
            self.adj_s = self.edge_pruning(self.adj_s)
        if reweight:
            # print("start reweight")
            self.adj_norm = self.edge_reweighting(self.adj)
            self.adj_a_norm = self.edge_reweighting(self.adj_a)
            self.adj_s_norm = self.edge_reweighting(self.adj_s)
        else:
            self.adj_norm = self.normalize_adj(self.adj)
            self.adj_a_norm = self.normalize_adj(self.adj_a)
            self.adj_s_norm = self.normalize_adj(self.adj_s)

    def get_adjs_norm(self, adj_names=('s', 'o', 'a')):
        adjs_norm = []
        if 's' in adj_names:
            adjs_norm.append(self.adj_s_norm)
        if 'o' in adj_names:
            adjs_norm.append(self.adj_norm)
        if 'a' in adj_names:
            adjs_norm.append(self.adj_a_norm)
        
        return adjs_norm

    def edge_pruning(self, adj):
        row, col = adj.nonzero()
        adj.eliminate_zeros()
        kl_loss = entropy(self.proba[row], self.proba[col], axis=1)
        edge_pruned = np.where(kl_loss < self.tau, 1.0, 0.0)
        adj[row,  col] = edge_pruned
        adj.eliminate_zeros()
        return adj

    def edge_reweighting(self, adj):
        edge_adj = adj
        edge_adj.eliminate_zeros()
        row, col = edge_adj.nonzero()
        edge_sim = self.sim_matrix[row, col]
        # edge_adj[row, col] = edge_sim
        edge_adj.data = edge_sim
        edge_adj.setdiag(0)
        edge_adj = normalize(edge_adj, axis=1, norm='l1')
        degree = (edge_adj != 0).sum(1).A1
        lam = 1 / (degree + 1)

        degree_rate = (degree / (degree + 1)).reshape(-1, 1)
        edge_adj = edge_adj.multiply(degree_rate)
        edge_adj.setdiag(lam)
        edge_adj.eliminate_zeros()
        return edge_adj

    def normalize_adj(self, mx):
        """Normalize sparse adjacency matrix,
        A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        Row-normalize sparse matrix

        Parameters
        ----------
        mx : scipy.sparse.csr_matrix
            matrix to be normalized

        Returns
        -------
        scipy.sprase.lil_matrix
            normalized matrix
        """
        if type(mx) is not sp.lil.lil_matrix:
            mx = mx.tolil()
        if mx[0, 0] == 0 :
            mx = mx + sp.eye(mx.shape[0])
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)
        return mx