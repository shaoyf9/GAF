import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

from GAF.gaf import GraphRefine
from GAF import utils_
from GAF.model import MGCN


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.01,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')

parser.add_argument('--k', type=int, default=15,  help='k for ang')
parser.add_argument('--sth', type=float, default=0.01,  help="sng for ang")
parser.add_argument('--tau', type=float, default=2.0,  help='tau for sng')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
# idx = (idx_train, idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

def tensor_to_csr(data):
    return sp.csr_matrix(data.cpu().numpy())

def test_gref(adj, features, labels, idx_train, idx_val, idx_test):
    ''' test on gref '''
    adj, features, labels = tensor_to_csr(adj), tensor_to_csr(features), labels.numpy()

    print(args.k, args.sth, args.tau)
    graph_ref = GraphRefine(adj, features, labels, idx_train, k=args.k, sth=args.sth, tau=args.tau)
    graph_ref.data_refine(reweight=True, prune=True)
    # choosen graph for Joint GCN
    aux_graphs = ['s', 'o', 'a']
    adjs_norm = graph_ref.get_adjs_norm(aux_graphs)

    for adj in adjs_norm:
        print("edges nums: {}".format(adj.nnz))

    # Convert data to tensor
    adjs_norm, features, labels = utils_.covert_to_tensor(adjs_norm, features, labels, device)

    result = []
    # Setup Joint GCN Model
    mgcn = MGCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()+1), device=device)
    mgcn = mgcn.to(device)

    mgcn.fit(features=features, labels=labels, idx_train=idx_train, 
                idx_val=idx_val, train_iters=200, verbose=True, patience=201, adjs=adjs_norm)

    mgcn.eval()
    output = mgcn.test(idx_test)
    print("==================================================")
    print("Test set results:",
          "accuracy= {:.4f}".format(output.cpu().item()))
def test_gcn(adj, features, labels, idx_train, idx_val, idx_test):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    # gcn.fit(features, adj, labels, idx_train) # train without model picking
    gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("==================================================")
    print("Test set on GCN results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def main():
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    
    modified_adj = model.modified_adj
    test_gcn(modified_adj, features, labels, idx_train, idx_val, idx_test)
    test_gref(modified_adj, features, labels, idx_train, idx_val, idx_test)

    modified_adj = sp.csr_matrix(modified_adj.cpu().numpy())

    # # if you want to save the modified adj/features, uncomment the code below
    # file_name = './data/{dataset}_meta_adj_{ptb_rate}.npz'.format(dataset=args.dataset, ptb_rate=args.ptb_rate)
    # sp.save_npz(file_name, modified_adj)

if __name__ == '__main__':
    main()