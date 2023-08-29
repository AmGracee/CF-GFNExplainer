# Based on https://github.com/tkipf/pygcn/blob/master/pygcn/train.py

from __future__ import division
from __future__ import print_function

import pickle
import sys
sys.path.append('../..')
import argparse
import numpy as np
import scipy.sparse as sp
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from gcn_node import GCN
from utils.utils import load_cora_data, accuracy_cora, normalize_adj,load_npz
from torch_geometric.utils import accuracy, dense_to_sparse, to_undirected, to_dense_adj
from load_dataset import *

# Defaults based on GNN Explainer
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--dataset', default='syn1')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--clip', type=float, default=2.0, help='Gradient clip).')

args = parser.parse_args()
args.cuda = True


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
if args.dataset == 'syn1':
    with open("../data/synthetic/{}.pickle".format(args.dataset[:4]), "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])
    args.dropout=0.0

elif args.dataset == 'cora':
    adj, features, labels, idx_train, idx_val, idx_test = load_cora_data()
    adj = adj.to_dense()
    args.dropout = 0.5

elif args.dataset == 'citeseer':
    adj, features, labels, idx_train, idx_test, valide_nodes = load_citeseer()
    args.epoch = 1000
    args.lr = 0.0001
    args.dropout = 0.5


norm_adj = normalize_adj(adj)
# Model and optimizer
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden, nclass=len(labels.unique()), dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.to(device)
    features = features.to(device)
    norm_adj = norm_adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    # idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, norm_adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    y_pred = torch.argmax(output, dim=1)
    acc_train = accuracy_cora(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, norm_adj)

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'time: {:.4f}s'.format(time.time() - t))

def test():
	model.eval()
	output = model(features, norm_adj)
	loss_test = F.nll_loss(output[idx_test], labels[idx_test])
	y_pred = torch.argmax(output, dim=1)
	acc_test = accuracy(y_pred[idx_test], labels[idx_test])
	print("Test set results:",
		  "loss= {:.4f}".format(loss_test.item()),
		  "accuracy= {:.4f}".format(acc_test))
	return y_pred

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

torch.save(model.state_dict(), "../models/gcn_3layer_dropout{}_{}_lr{}_hid{}_weightdecay{}_epoch{}_removeisolated".format(args.dropout, args.dataset,args.lr,args.hidden,args.weight_decay,args.epoch) + ".pt")

# Testing
y_pred = test()

print("y_true counts: {}".format(np.unique(labels.cpu().numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(np.unique(y_pred.cpu().numpy(), return_counts=True)))
print("Finished training!")