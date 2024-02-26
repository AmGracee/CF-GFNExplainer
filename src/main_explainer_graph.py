# from __future__ import division
# from __future__ import print_function
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('..')
import argparse
import pickle # 持续化模块：就是让数据持久化保存
import torch.optim as optim

from numpy.random import default_rng

from gcn_graph import GCN_graph
from utils.utils import normalize_adj, safe_open, normalize_adj_cuda
from utils.load_dataset import *

from cfgflownet.env_graph import GFlowNetEnv
from cfgflownet.replay_buffer import ReplayBuffer
from cfgflownet.gflownet_graph import GFlowNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mutag')

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--nhops', type=int, default=3, help='nhop of subgraph.')
parser.add_argument('--beta', type=int, default=0)
parser.add_argument('--transformer_num_layers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)

# For Environment
parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments.')

# For Replay buffer
parser.add_argument('--replay_capacity', type=int, default=16800, help='Capacity of the replay buffer.')
parser.add_argument('--prefill', type=int, default=100, help='Number of iterations with a random policy to prefill')#500
parser.add_argument('--num_iterations', type=int, default=500, help='Number of iterations')#1000

# For Exploration
parser.add_argument('--min_exploration', type=float, default=0.1, help='Minimum value of epsilon-exploration')


# For Optimization
parser.add_argument('--delta', type=float, default=0.5, help='Value of delta for Huber loss')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adam')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
# print('args.device'.format(args.device))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
gen = torch.default_generator
rng = default_rng(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(args)

adj, features, labels, idx_train, idx_test, valid_graph_idx = load_mutag()
nfeat = features[0].shape[1]
nclass = len(labels.unique())


# Set up original model, get predictions
gcn_model = GCN_graph(nfeat, args.hidden, nclass, device, args.dropout)
gcn_model.load_state_dict(torch.load("../models/gcn_3layer_dropout0.1_mutag.pt",map_location=device))
gcn_model.eval()
gcn_model.to(device)
print(gcn_model.conv1.bias)
print(valid_graph_idx)
y_pred_orig = []
for graph_idx in valid_graph_idx:
    sub_adj = adj[graph_idx]
    sub_feat = features[graph_idx]
    sub_labels = labels[graph_idx]
    if args.cuda:
        gcn_model.to(device)
        sub_adj = sub_adj.to(device)
        sub_feat = sub_feat.to(device)
    norm_sub_adj = normalize_adj_cuda(sub_adj)
    sub_output = gcn_model(sub_feat, norm_sub_adj)
    sub_output = sub_output.squeeze()
    sub_pred_orig = torch.argmax(sub_output, dim=0)
    y_pred_orig.append(sub_pred_orig.item())
    # sub_output_orig = torch.softmax(sub_output, dim=0)[sub_pred_orig]

# Get CF examples in test set
actions_batch = []
test_cf_examples,batch_terminal_state,batch_hard_node,num_diversities, diversities = [],[],[],[],[]
spent_times,edges_acc_top5 = [],[]
for graph_idx in valid_graph_idx:
    # print(graph_idx)
    sub_adj = adj[graph_idx]
    sub_feat = features[graph_idx]
    sub_labels = labels[graph_idx]

    num_edges_orig = sum(sum(sub_adj)) / 2

    if args.cuda:
        gcn_model.to(device)
        sub_adj = sub_adj.to(device)
        sub_feat = sub_feat.to(device)

    norm_sub_adj = normalize_adj_cuda(sub_adj)
    sub_output = gcn_model(sub_feat, norm_sub_adj)
    sub_output = sub_output.squeeze()
    sub_pred_orig = torch.argmax(sub_output, dim=0)
    sub_output_orig = torch.softmax(sub_output, dim=0)[sub_pred_orig]

    reward_orig = torch.tensor(0.).to(device)

    env = GFlowNetEnv(
        sub_adj=sub_adj,
        num_envs=args.num_envs,
        reward_orig=reward_orig,
        sub_output_orig=sub_output_orig,
    )

    # Create the replay buffer, push trajectory into it.
    replay = ReplayBuffer(
        args.replay_capacity,
        sub_adj=sub_adj)
    observations = env.reset()

    # Create the GFlowNet & initialize parameters
    mask = torch.zeros_like(sub_adj)
    gflownet = GFlowNet(gcn_model=gcn_model,
                        sub_adj=sub_adj,
                        sub_feat=sub_feat,
                        sub_pred_orig=sub_pred_orig,
                        num_edges_orig=num_edges_orig,
                        y_pred_orig=sub_pred_orig,
                        mask=mask,
                        env=env,
                        replay=replay,
                        observations=observations,
                        device=device,
                        delta=args.delta)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(gflownet.GFN_model.parameters(),lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(gflownet.GFN_model.parameters(),lr=args.lr)

    gflownet.GFN_model.to(device)

    env._state['sub_adj'] = env._state['sub_adj'].to(device)
    env._state['mask'] = env._state['mask'].to(device)
    observations['sub_adj'] = observations['sub_adj'].to(device)
    observations['mask'] = observations['mask'].to(device)

    cf_example, terminal_state, fidelity_probs, diversity,spent_time = gflownet.explain(
                                   graph_idx=graph_idx,
                                   optimizer=optimizer,
                                   prefill=args.prefill,
                                   num_iterations=args.num_iterations,
                                   min_exploration=args.min_exploration,
                                   batch_size=args.batch_size,
                                   indices=None,
                                   rng=rng)

    if len(cf_example) < 1:
        continue

    if cf_example != []:
        cf_example.append(fidelity_probs)
        cf_example.append(spent_time)
        spent_times.append(spent_time)
        diversities.append([graph_idx,diversity])

    test_cf_examples.append([cf_example])
    batch_terminal_state.append(terminal_state)

    torch.save(gflownet.GFN_model.state_dict(),
               "../models/gflownet/{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchszie{}_graph_idx{}".format(
                   args.dataset,args.dataset, args.nhops, args.num_envs, args.delta, args.prefill, args.num_iterations,
                   args.transformer_num_layers,
                   args.batch_size, graph_idx) + ".pt")

    with safe_open("../results/{}/gflownet/iter{}/tmp/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}_graphidx{}".format(
                    args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs,
                    args.delta, args.prefill, args.num_iterations,
                    args.transformer_num_layers, args.batch_size, graph_idx) + "_best_cf_example", "wb") as f:
        pickle.dump(test_cf_examples, f)

    with safe_open("../results/{}/gflownet/iter{}/tmp/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}_graphidx{}".format(
                    args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs,
                    args.delta, args.prefill, args.num_iterations,
                    args.transformer_num_layers, args.batch_size, graph_idx) + "_diversities", "wb") as f:
        pickle.dump(diversities, f)
print(spent_times)
print("Total time elapsed: {:.4f}min".format(sum(spent_times)))
print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(valid_graph_idx)))

torch.save(gflownet.GFN_model.state_dict(),"../models/gflownet/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchszie{}".format(
               args.dataset, args.nhops, args.num_envs, args.delta, args.prefill, args.num_iterations,
               args.transformer_num_layers,
               args.batch_size) + ".pt")

with safe_open("../results/{}/gflownet/iter{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}".format(
                args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs, args.delta,
                args.prefill, args.num_iterations,
                args.transformer_num_layers, args.batch_size) + "_best_cf_example_54-187", "wb") as f:
    pickle.dump(test_cf_examples, f)

with safe_open("../results/{}/gflownet/iter{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}".format(
                args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs, args.delta,
                args.prefill, args.num_iterations,
                args.transformer_num_layers, args.batch_size) + "_diversities_54-187", "wb") as f:
    pickle.dump(diversities, f)

with safe_open("../results/{}/gflownet/iter{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}_trajectoies".format(
                args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs, args.delta,
                args.prefill, args.num_iterations,
                args.transformer_num_layers, args.batch_size) + '_terminal_state_54-187', "wb") as f:
    pickle.dump(batch_terminal_state, f)
print(' ')
