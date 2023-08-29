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

from gcn_node import GCN
from utils.utils import normalize_adj, get_neighbourhood, safe_open, normalize_adj_cuda
from utils.load_dataset import *

from cfgflownet.env_node import GFlowNetEnv
from cfgflownet.replay_buffer import ReplayBuffer
from cfgflownet.gflownet import GFlowNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='citeseer')

# Based on original GCN models
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--nhops', type=int, default=3, help='nhop of subgraph.')
parser.add_argument('--beta', type=int, default=0, help='whether use gflownet.')
parser.add_argument('--transformer_num_layers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)

# For Environment
parser.add_argument('--num_envs', type=int, default=8, help='Number of parallel environments.')

# For Replay buffer
parser.add_argument('--replay_capacity', type=int, default=4800, help='Capacity of the replay buffer.')
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
np.random.seed(args.seed)
torch.manual_seed(args.seed)
gen = torch.default_generator
rng = default_rng(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if args.dataset == 'syn1':
    adj, features, labels, idx_train, idx_test, valid_nodes = load_syn1()
    args.nhops = 4
    args.dropout = 0.0
elif args.dataset == 'syn4':
    adj, features, labels, idx_train, idx_test, valid_nodes = load_syn4_addedges()
    args.nhops = 4
    args.dropout = 0.0
elif args.dataset == 'syn5':
    adj, features, labels, idx_train, idx_test,valid_nodes = load_syn5_addedges()
    args.nhops = 4
    args.dropout = 0.0
elif args.dataset == 'cora':
    adj,features,labels,idx_train, idx_test,valid_nodes = load_cora()
    args.nhops = 3
    args.dropout = 0.5
elif args.dataset == 'citeseer':
    adj, features, labels, idx_train, idx_test, valid_nodes = load_citeseer()
    args.nhops = 3
    args.dropout = 0.5

print(args)

edge_index = dense_to_sparse(adj)
# adj = to_dense_adj(edge_index)
norm_adj = normalize_adj(adj)
# Set up original model, get predictions
gcn_model = GCN(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,nclass=len(labels.unique()), dropout=args.dropout)

gcn_model.load_state_dict(torch.load("../models/gcn_3layer_dropout{}_{}.pt".format(args.dropout, args.dataset),map_location=device))

gcn_model.eval()  # 训练完train_datasets之后，model要来测试样本了。在model(test_datasets)之前，需要加上model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值
gcn_model.to(device)
print(gcn_model.gc1.bias)
output = gcn_model(features.to(device), norm_adj.to(device))
y_pred_orig = torch.argmax(output, dim=1)

actions_batch = []
test_cf_examples,batch_terminal_state,batch_hard_node, diversities = [],[],[],[]
spent_times = []
print('The number of valid instance:{}'.format(valid_nodes.shape[0]))
print(valid_nodes)
for i in valid_nodes[:]:

    sub_adj, sub_feat, sub_labels, node_dict, _ = get_neighbourhood(int(i), edge_index, args.nhops, features, labels)

    new_idx = node_dict[int(i)]
    num_edges_orig = sum(sum(sub_adj)) / 2

    if args.cuda:
        gcn_model.to(device)
        sub_adj = sub_adj.to(device)
        sub_feat = sub_feat.to(device)
        output = output.to(device)
        y_pred_orig = y_pred_orig.to(device)

    norm_sub_adj = normalize_adj_cuda(sub_adj)
    sub_output = gcn_model(sub_feat, norm_sub_adj)
    sub_pred_orig = torch.argmax(sub_output, dim=1)[new_idx]
    sub_output_orig = torch.softmax(sub_output[new_idx],dim=0)[sub_pred_orig]


    with torch.no_grad():
        print("Output original model, full adj: {}".format(output[i]))
        print("Output original model, sub adj: {}".format(gcn_model(sub_feat, normalize_adj_cuda(sub_adj))[new_idx]))

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
                        node_dict=node_dict,
                        y_pred_orig=y_pred_orig,
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
                                   node_idx=i,
                                   new_idx=new_idx,
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
        diversities.append([i.item(),diversity])

    test_cf_examples.append([cf_example])
    batch_terminal_state.append(terminal_state)

    torch.save(gflownet.GFN_model.state_dict(),
               "../models/gflownet/{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchszie{}_node_idx{}".format(
                   args.dataset,args.dataset, args.nhops, args.num_envs, args.delta, args.prefill, args.num_iterations,
                   args.transformer_num_layers,
                   args.batch_size, i.item()) + ".pt")

    with safe_open(
            "../results/{}/gflownet/iter{}/tmp/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}_nodeidx{}".format(
                    args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs,
                    args.delta, args.prefill, args.num_iterations,
                    args.transformer_num_layers, args.batch_size, i.item()) + "_best_cf_example", "wb") as f:
        pickle.dump(test_cf_examples, f)

    with safe_open(
            "../results/{}/gflownet/iter{}/tmp/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}_nodeidx{}".format(
                    args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs,
                    args.delta, args.prefill, args.num_iterations,
                    args.transformer_num_layers, args.batch_size, i.item()) + "_diversities", "wb") as f:
        pickle.dump(diversities, f)

print(spent_times)
print("Total time elapsed: {:.4f}min".format(sum(spent_times)))
print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(valid_nodes)))

torch.save(gflownet.GFN_model.state_dict(),
           "../models/gflownet/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchszie{}".format(
               args.dataset, args.nhops, args.num_envs, args.delta, args.prefill, args.num_iterations,
               args.transformer_num_layers,
               args.batch_size) + ".pt")

with safe_open(
        "../results/{}/gflownet/iter{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}".format(
                args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs, args.delta,
                args.prefill, args.num_iterations,
                args.transformer_num_layers, args.batch_size) + "_best_cf_example", "wb") as f:
    pickle.dump(test_cf_examples, f)

with safe_open(
        "../results/{}/gflownet/iter{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}".format(
                args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs, args.delta,
                args.prefill, args.num_iterations,
                args.transformer_num_layers, args.batch_size) + "_diversities", "wb") as f:
    pickle.dump(diversities, f)

with safe_open(
        "../results/{}/gflownet/iter{}/{}_gflownet_nhops{}_numenvs{}_delta{}_prefill{}_iterations{}_trans{}_batchsize{}_trajectoies".format(
                args.dataset, args.prefill + args.num_iterations, args.dataset, args.nhops, args.num_envs, args.delta,
                args.prefill, args.num_iterations,
                args.transformer_num_layers, args.batch_size) + '_terminal_state', "wb") as f:
    pickle.dump(batch_terminal_state, f)
print(' ')