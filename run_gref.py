import torch
import numpy as np
from GREF.gref import GraphRefine
from GREF import utils_
from GREF.model import MGCN
from deeprobust.graph.utils import *
from deeprobust.graph import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta', choices=['meta', 'nettack', 'random'], help='attack method')
parser.add_argument('--ptb_rate', type=float, default=0.25,  help='pertubation rate')
parser.add_argument('--run_times', type=int, default=10, help='run times of GREF')
parser.add_argument('--aux_graphs', nargs='+', default=['o', 's', 'a'], help='graph for GCN, s:structural neighborhood graph, o:orignal graph, a:attributive neighborhood graph')
                    
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

if args.dataset == "cora":
    k, sth, tau = 15, 0.01, 2.0
elif args.dataset == "citeseer":
    k, sth, tau = 15, 0.01, 1.25
elif args.dataset == "pubmed":
    k, sth, tau =  15, 0.4, 0.25

# Data refinement
graph_ref = GraphRefine(perturbed_adj, features, labels, idx_train, k=15, sth=sth, tau=tau)
graph_ref.data_refine(reweight=True, prune=True)
# choosen graph for Joint GCN
aux_graphs = args.aux_graphs
adjs_norm = graph_ref.get_adjs_norm(aux_graphs)

for adj in adjs_norm:
    print("edges nums: {}".format(adj.nnz))

# Convert data to tensor
adjs_norm, features, labels = utils_.covert_to_tensor(adjs_norm, features, labels, device)

result = []
for i in range(args.run_times):
    # Setup Joint GCN Model
    print('=================start {}-th train Joint GCN================='.format(i + 1))
    model = MGCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()+1), device=device)
    model = model.to(device)

    model.fit(features=features, labels=labels, idx_train=idx_train, 
                idx_val=idx_val, train_iters=200, verbose=True, patience=201, adjs=adjs_norm)

    model.eval()
    output = model.test(idx_test)
    result.append(output.cpu().item())
result = np.array(result)
print("run {} times: accuracy mean: {}, accuracy std: {}".format(args.run_times, result.mean(), result.std()))