import torch
import pickle
import pandas as pd
import numpy as np
from utils.load_dataset import *




header = ['node_idx','new_idx','cf_adj','pred_orig','cf_pred','num_removed_edges','order','fidelity_probs','time']
path_cfexample = '../results/syn5/gflownet/iter1600/syn5_gflownet_nhops4_numenvs8_delta0.5_prefill100_iterations1500_trans1_batchsize4_best_cf_example'
path_diversity = '../results/syn5/gflownet/iter1600/syn5_gflownet_nhops4_numenvs8_delta0.5_prefill100_iterations1500_trans1_batchsize4_diversities'
# path_terminal = '../results/syn5/gflownet/iter2100/syn5_gflownet_nhops4_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_trajectoies_terminal_state'


hidden = 20
dropout = 0.0

if "syn4" in path_cfexample:
	dataset = "syn4"
elif "syn5" in path_cfexample:
	dataset = "syn5"
elif "syn1" in path_cfexample:
	dataset = "syn1"
elif "cora" in path_cfexample:
	dataset = "cora"
elif "citeseer" in path_cfexample:
	dataset = "citeseer"


#加载数据集
if dataset == 'cora':
    adj,features,labels,idx_train, idx_test, valid_nodes = load_cora()
    nhops = 3
    dropout = 0.5
elif dataset == 'citeseer':
    adj, features, labels, idx_train, idx_test, valid_nodes = load_citeseer()
elif dataset == 'syn5':
    adj, features, labels, idx_train, idx_test, valid_nodes = load_syn5_addedges()
    nhops = 4
    dropout = 0.0
elif dataset == 'syn4':
    adj, features, labels, idx_train, idx_test, valid_nodes = load_syn4_addedges()
    nhops = 4
    dropout = 0.0

#
# with open(path_terminal,'rb') as f3:
#     terminal = pickle.load(f3)
#     terminal_state_935 = terminal
#     print('')

with open(path_cfexample, 'rb') as f:
    trajectories = pickle.load(f)
    df_prep = []
    for trajectory in trajectories:
        if trajectory == []:
            continue
        for traj in trajectory:
            df_prep.append(traj)
    df = pd.DataFrame(df_prep, columns=header)


diversities_new = []
df_num_diversity=[]
with open(path_diversity,'rb') as f2:
    diversities = pickle.load(f2)
    for diversity in diversities:
        df_num_diversity.append(len(diversity[1]))
        diversity_new = {}
        node_idx = diversity[0]
        for i in range(len(diversity[1])):
            perturb = diversity[1][i][0]
            if len(perturb) in diversity_new.keys():
                diversity_new[len(perturb)].append(perturb) #diversity_new[删边数量]=perturb
            else:
                diversity_new[len(perturb)] = [perturb]
        diversities_new.append([node_idx,diversity_new])

df["num_diversity"] = df_num_diversity

df.to_excel("../results/syn5/gflownet/iter1600/df_syn5_nhops4_numenvs8_prefill100_iter1500_trans1_batsize4_best_cf_example.xlsx", sheet_name="Sheet1", index=False)

print('')


def encode(sub_adj, decoded):
    encoded = decoded.reshape(-1, sub_adj.shape[0] ** 2)
    encoded = encoded.cpu().numpy().astype('int16')
    encoded = np.packbits(encoded, axis=1)
    return encoded


def decode(sub_adj, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1,count=sub_adj.shape[0] ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], sub_adj.shape[0],sub_adj.shape[0])
        return decoded.astype(dtype)



