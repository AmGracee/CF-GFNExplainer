import torch
import pickle
import pandas as pd
import numpy as np
from utils.load_dataset import *





header = ['graph_idx','cf_adj','pred_orig','cf_pred','num_removed_edges','order','fidelity_probs','time']
path_cfexample = '../results/mutag/gflownet/iter1600/mutag_gflownet_nhops3_numenvs8_delta0.5_prefill100_iterations1500_trans1_batchsize4_best_cf_example'
path_diversity = '../results/mutag/gflownet/iter1600/mutag_gflownet_nhops3_numenvs8_delta0.5_prefill100_iterations1500_trans1_batchsize4_diversities'
# path_terminal = '../results/mutag/gflownet/iter2100/mutag_gflownet_nhops3_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_trajectoies_terminal_state'

with open(path_cfexample, 'rb') as f:
    trajectories = pickle.load(f)
    df_prep = []
    for trajectory in trajectories:
        if trajectory == []:
            continue
        for traj in trajectory:
            #decode adj
            # traj['terminal_state'] = decode(traj['terminal_state'])
            # traj[2] = torch.tensor(decode(traj[2])).squeeze(0)
            df_prep.append(traj)
    df = pd.DataFrame(df_prep, columns=header)

diversities_new = []
df_num_diversity=[]
with open(path_diversity,'rb') as f2:
    diversities = pickle.load(f2)#diversity[-1]表示哪些有重复
    for diversity in diversities:  # diversity 存放的都是不重叠的cf_adj
        df_num_diversity.append(len(diversity[1]))
        diversity_new = {}
        node_idx = diversity[0]
        for i in range(len(diversity[1])):
            perturb = diversity[1][i][0]
            if len(perturb) in diversity_new.keys():
                diversity_new[len(perturb)].append(perturb)  # diversity_new[删边数量]=perturb
            else:
                diversity_new[len(perturb)] = [perturb]
        diversities_new.append([node_idx, diversity_new])

df["num_diversity"] = df_num_diversity

df.to_excel("../results/mutag/gflownet/iter1600/df_syn5_nhops4_numenvs8_prefill100_iter1500_trans1_batsize4_best_cf_example.xlsx", sheet_name="Sheet1", index=False)
print(' ')



def encode(sub_adj, decoded):
    encoded = decoded.reshape(-1, sub_adj.shape[0] ** 2)
    encoded = encoded.cpu().numpy().astype('int16')
    encoded = np.packbits(encoded, axis=1)
    return encoded


def decode(sub_adj, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1,count=sub_adj.shape[0] ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], sub_adj.shape[0],sub_adj.shape[0])
        return decoded.astype(dtype)






