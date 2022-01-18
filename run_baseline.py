import torch
import numpy as np
from GAF.gaf import GraphRefine
from GAF import utils_
from deeprobust.graph.utils import *
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN, GCNJaccard, GCNSVD, RGCN, ProGNN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta', choices=['meta', 'nettack', 'random'], help='attack method')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')
parser.add_argument('--run_times', type=int, default=10, help='run times of GAF')
parser.add_argument('--defence', type=str, default='gcn', choices=['gcn', 'gcn-jaccard', 'gcn-svd', 'rgcn', 'prognn'])
# threshold for GCN-jaccard
parser.add_argument('--td', type=float, default=0.01,  help='threshold')
# Truncated components for GCN-SVD
parser.add_argument('--svdk', type=int, default=100, help='Truncated Components.')
# hidden units for RGCN
parser.add_argument('--nhid', type=int, default=128, help='hidden units.')

# args for prognn
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load preattacked data
data, idx = utils_.load_preattacked_data(args.dataset, args.attack, args.ptb_rate, path="./data/")
adj, perturbed_adj, features, labels = data
idx_train, idx_val, idx_test = idx

# Setup GCN Model
if args.defence == "gcn":
    model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device, lr=0.01, dropout=0.5)
    model = model.to(device)
    model.fit(features, perturbed_adj, labels, idx_train, idx_val=idx_val, train_iters=200, verbose=True, patience=25)
    model.eval()
elif args.defence == "gcn-jaccard":
    isbinary = np.array_equal(features, features.astype(bool))
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device, binary_feature=isbinary)
    model = model.to(device)
    threshold = args.td
    print('=== testing GCN-Jaccard on perturbed graph ===')
    model.fit(features.toarray(), perturbed_adj, labels, idx_train, idx_val, threshold=threshold, train_iters=200)
elif args.defence == 'gcn-svd':
    model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device)
    model = model.to(device)
    k = args.svdk
    print('=== testing GCN-SVD on perturbed graph ===')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=k, verbose=True,train_iters=200)
elif args.defence == 'rgcn':
    nhid = args.nhid
    model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
            nhid=nhid, device=device)
    model = model.to(device)
    print('=== testing RGCN on perturbed graph ===')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
elif args.defence == "prognn":
    gcn_model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, device=device)

    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)

    model = ProGNN(gcn_model, args, device)
    model.fit(features, perturbed_adj, labels, idx_train, idx_val)
    model.test(features, labels, idx_test)
 
# You can use the inner function of model to test
if args.defence != "prognn":
    output = model.test(idx_test)