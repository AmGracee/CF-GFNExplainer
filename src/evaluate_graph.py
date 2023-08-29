import torch
import pickle
import pandas as pd
import numpy as np
from torch_geometric.utils import dense_to_sparse
# from src.utils.utils import load_cora_data, normalize_adj, get_neighbourhood
from src.utils.utils import load_cora_data, get_neighbourhood
from gcn_graph import GCN_graph
from utils.utils import normalize_adj, get_neighbourhood, safe_open,load_cora_data,encode,decode,edges_unique,new_idx_edge_into_old_idx,old_idx_edge_into_new_idx
from utils.load_dataset import *




# 查看trajectory
header = ['graph_idx','cf_adj','pred_orig','cf_pred','num_removed_edges','order','fidelity_probs','num_diversity','time']
path_cfexample = '../results/mutag/gflownet/iter2100/mutag_gflownet_nhops3_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_best_cf_example'
path_diversity = '../results/mutag/gflownet/iter2100/mutag_gflownet_nhops3_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_diversities'
path_terminal = '../results/mutag/gflownet/iter2100/mutag_gflownet_nhops3_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_trajectoies_terminal_state'
# path1 = '../results/syn1/gflownet/iter2100/tmp/syn1_gflownet_nhops4_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_nodeidx227_best_cf_example'
# path2 = '../results/syn1/gflownet/iter2100/syn1_gflownet_nhops4_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_best_cf_example'
# with open(path2, 'rb') as f2:
#     trajectories2 = pickle.load(f2)
# with open(path1, 'rb') as f1:
#     trajectories1 = pickle.load(f1)
# traj =  trajectories1+trajectories2
# with safe_open("../results/syn1/gflownet/iter2100/syn1_gflownet_nhops4_numenvs8_delta0.5_prefill100_iterations2000_trans1_batchsize4_best_cf_example1", "wb") as f3:
#     pickle.dump(traj, f3)

np.random.seed(42)
torch.manual_seed(42)
hidden = 32
dropout = 0.1

dataset = "mutag"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#加载数据集
adj, features, labels, idx_train, idx_test, valid_graph_idx = load_mutag() #shuffle了，注意写随机种子


gcn_model = GCN_graph(features[0].shape[1], hidden, len(labels.unique()), device, dropout)
gcn_model.load_state_dict(torch.load("../models/gcn_3layer_dropout0.1_mutag_lr0.001_hid32_weightdecay0.001_epoch500_noshuffle.pt"))
gcn_model.eval()

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

with open(path_diversity,'rb') as f2:
    diversity = pickle.load(f2)#diversity[-1]表示哪些有重复
    print('')


# with open(path_terminal,'rb') as f3:
#     terminal = pickle.load(f3)
#     terminal_state_935 = terminal[188]
#     print('')
#
# for nodeidx,new,terminal,cf_indica,ord in terminal_state_935:
#     node_idx = terminal_state_935[nodeidx]

# num_edges = sum(sum(df['sub_adj'])).item()
# for i in df.index:
#     num_edges.append(sum(sum(df["sub_adj"][i])))
# df["num_edges"] = num_edges

# df.to_excel("../results/syn5/gflownet/iter1100/df_syn5_nhops4_numenvs8_prefill100_iter1000_trans1_batsize4_best_cf_example.xlsx", sheet_name="Sheet1", index=False)
print(' ')
df_motif = df[df["pred_orig"] != 0].reset_index(drop=True)
num_edges,sub_adjs,cf_adjs = [],[],[]
for i in df.index:
    sub_adj = get_neighbourhood(int(df['node_idx'][i]),edge_index,4,features,labels)[0]
    num_node = sub_adj.shape[0]
    cf_adj = torch.tensor(decode(num_node, df["cf_adj"][i])).squeeze()
    num_edges.append(sub_adj.sum().item() / 2)
df["num_edges"] = num_edges



accuracy = []
full_node_idx = sorted(np.concatenate((idx_train.numpy(), idx_test.numpy())))
dict_ypred_orig = {}
for i in full_node_idx:
    sub_adj,sub_feat,sub_labels,node_dict,_ = get_neighbourhood(int(i),edge_index,4,features,labels)
    new_idx = node_dict[int(i)]
    norm_sub_adj = normalize_adj(sub_adj)
    dict_ypred_orig[i] = torch.argmax(model(sub_feat,norm_sub_adj)[new_idx]).item()

for i in range(len(df_motif)): #df_motif:在house上的节点的反事
    i = 25
    node_idx = df_motif["node_idx"][i]
    new_idx = df_motif["new_idx"][i]
    cf_pred_node = df_motif["cf_pred"][i]
    pred_orig_node = df_motif["pred_orig"][i]
    sub_adj, sub_feat, _, node_dict,_ = get_neighbourhood(int(node_idx), edge_index, 4, features, labels)
    sub_output = model(sub_feat, normalize_adj(torch.tensor(sub_adj)))
    pred = torch.argmax(sub_output, dim=1)
    num_node = sub_adj.shape[0]
    # print("node_dict:", node_dict)  # 原始预测节点的index映射到新的index 格式={xx:xx}
    # Confirm idx mapping is correct
    if node_dict[node_idx] == df_motif["new_idx"][i]: #判断取3阶邻居的新节点index 是否等于 cf中的新节点index
        cf_adj = df_motif["cf_adj"][i]
        cf_adj = decode(num_node, cf_adj).squeeze()
        cf_output = model(sub_feat, normalize_adj(torch.tensor(cf_adj)))
        cf_pred = torch.argmax(cf_output, dim=1)
        # sub_adj = df_motif["sub_adj"][i]
        # sub_adj = decode(num_node, sub_adj)
        perturb = np.abs(cf_adj - sub_adj.numpy())
        perturb_edges = np.nonzero(perturb)  # Edge indices nonzero得到perturb中非0元素的index，即得到扰动边的位置的index
        perturb_edges_unique = edges_unique(list(zip(perturb_edges[0],perturb_edges[1])))#边去重
        perturb_edges = new_idx_edge_into_old_idx(perturb_edges_unique,node_dict)

        # Retrieve original node idxs for original predictions
        edges_in_motif_count = 0
        perturb_edges_pred,edges_in_motif = [],[]
        for j in perturb_edges:
            perturb_edges_pred.append((y_pred_orig[j[0]].item(),y_pred_orig[j[1]].item()))
            if y_pred_orig[j[0]] and y_pred_orig[j[1]]:
                edges_in_motif.append((j[0],j[1]))
                edges_in_motif_count += 1

        prop_correct = edges_in_motif_count / len(perturb_edges)
        # 反事实的正确率判断：
        accuracy.append([node_idx, new_idx, pred_orig_node, cf_pred_node,perturb_edges, edges_in_motif,perturb_edges_pred,prop_correct])


df_accuracy = pd.DataFrame(accuracy, columns=["node_idx", "new_idx", "pred_orig", "cf_pred","perturb_edges","edges_in_motif_old","perturb_edges_pred","prop_correct"])
df_accuracy_no100 = df_accuracy[df_accuracy['prop_correct'] < 1.].reset_index(drop=True)
indicator = []
for j in range(len(df_accuracy_no100)):
    node_idx = df_accuracy_no100["node_idx"][j]
    new_idx = df_accuracy_no100["new_idx"][j]
    edges_in_motif_old = df_accuracy_no100["edges_in_motif_old"][j]
    sub_adj, sub_feat, _, node_dict, _ = get_neighbourhood(int(node_idx), edge_index, 4, features, labels)
    edges_in_motif_new = old_idx_edge_into_new_idx(edges_in_motif_old, node_dict)
    for edge in edges_in_motif_new:
        sub_adj[edge[0]][edge[1]] = 0
        sub_adj[edge[1]][edge[0]] = 0
    perturb_adj_output = model(sub_feat, normalize_adj(sub_adj))
    perturb_pred = torch.argmax(perturb_adj_output, dim=1)[new_idx]
    if perturb_pred != y_pred_orig[node_idx]:
        indicator.append('yes')
    else:
        indicator.append('no')

#indicator是检查 在acc≠100%时，仅删除的在motif上的边 发现预测结果发生变化 为yes，发现gflownet找到的反事实不是最小的，但是最小的反事实包含在找到的反事实中
df_accuracy_no100['indicator'] = indicator

print(path)
print("Num cf examples found: {}/{}".format(len(df), len(idx_test)))
print("Accuracy", np.mean(df_accuracy["prop_correct"]), np.std(df_accuracy["prop_correct"]))
print(" ")
print("***************************************************************")
print(" ")



def encode(sub_adj, decoded):
    encoded = decoded.reshape(-1, sub_adj.shape[0] ** 2)
    encoded = encoded.cpu().numpy().astype('int16')
    encoded = np.packbits(encoded, axis=1)
    return encoded


def decode(sub_adj, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1,count=sub_adj.shape[0] ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], sub_adj.shape[0],sub_adj.shape[0])
        return decoded.astype(dtype)






