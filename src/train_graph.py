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
from utils.utils import load_cora_data, accuracy_cora, normalize_adj,normalize_adj_cuda,load_npz
from torch_geometric.utils import accuracy, dense_to_sparse, to_undirected, to_dense_adj
from load_dataset import *
from gcn_graph import GCN_graph
# Defaults based on GNN Explainer
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--dataset', default='mutag')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--clip', type=float, default=2.0, help='Gradient clip).')

args = parser.parse_args()
args.cuda = True


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
if args.dataset == 'mutag':
    adj, features, labels, idx_train,idx_test,valid_graph_idx = load_mutag()
elif args.dataset == 'bbbp':
    adj, features, labels, idx_train,idx_test,valid_graph_idx = load_bbbp_symm_save_pickle()
nfeat = features[0].shape[1]
nclass = len(torch.tensor(labels).unique())

model = GCN_graph(nfeat,args.hidden,nclass,device,dropout=args.dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    loss_all = 0
    for idx in idx_train:#每个图求一个output
        sub_feat = features[idx].to(device)
        sub_adj = adj[idx].to(device)
        norm_adj = normalize_adj_cuda(sub_adj)
        optimizer.zero_grad()
        output = model(sub_feat, norm_adj)
        loss_train = F.nll_loss(output, labels[idx].unsqueeze(0).to(device))
        loss_train.backward()
        loss_all += loss_train.item()
        optimizer.step()
    print('')
    return loss_all / len(idx_train)




def test(idxs):
    model.eval()
    y = torch.tensor([]).long().to(device)
    yp = torch.tensor([]).long().to(device)
    loss_all = 0
    for idx in idxs:
        sub_feat = features[idx].to(device)
        sub_adj = adj[idx].to(device)
        norm_adj = normalize_adj_cuda(sub_adj)
        output = model(sub_feat, norm_adj)
        loss = F.nll_loss(output, labels[idx].unsqueeze(0).to(device))
        pred = output.max(dim=1)[1]

        y = torch.cat([y, labels[idx].unsqueeze(0).to(device)])  # y is label
        yp = torch.cat([yp, pred])  # yp is pred

        loss_all += loss.item()
    acc = int((y == yp).sum()) / yp.numel()
    return acc, loss_all


# Train model
t_total = time.time()
best_acc = 0
for epoch in range(args.epochs):
    loss = train(epoch)
    train_acc, train_loss = test(idx_train)
    test_acc,test_loss = test(idx_test)
    print(f'Epoch: {epoch}, Loss: {loss:.5f}')
    print(f'Train -> Acc: {train_acc:.5f}')
    print(f'Test -> Acc: {test_acc:.5f}')

    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),
                   "../models/gcn_3layer_dropout{}_{}_lr{}_hid{}_weightdecay{}_epoch{}".format(args.dropout,
                                                                                               args.dataset, args.lr,
                                                                                               args.hidden,
                                                                                               args.weight_decay,
                                                                                               args.epochs) + ".pt")
        print("New best model saved!")

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# torch.save(model.state_dict(), "../models/gcn_3layer_dropout{}_{}_lr{}_hid{}_weightdecay{}_epoch{}".format(args.dropout, args.dataset,args.lr,args.hidden,args.weight_decay,args.epochs) + ".pt")

# Testing
# test_acc, _ = test(test_loader)

# print("y_true counts: {}".format(np.unique(labels.cpu().numpy(), return_counts=True)))
# print("y_pred_orig counts: {}".format(np.unique(y_pred.cpu().numpy(), return_counts=True)))
print("Finished training!")