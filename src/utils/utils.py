import os
import errno
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph,to_undirected
import scipy.sparse as sp




class linear_schedule(nn.Module):
    def __init__(self,init_value,end_value,transition_steps,transition_begin):
        super(linear_schedule, self).__init__()
        self.init_value = init_value
        self.end_value = end_value
        self.transition_steps = transition_steps
        self.transition_begin = transition_begin

    def forward(self,count):
        count = np.clip(count - self.transition_begin, 0, self.transition_steps)
        frac = 1 - count / self.transition_steps
        return (self.init_value - self.end_value) * (frac ** 1) + self.end_value

def encode_onehot(labels):
	classes = list(set(labels))
	classes.sort(key=list(labels).index)
	classes_dict = {c : np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
	return labels_onehot

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换成torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
        # tocoo将此矩阵转换成coo格式，astype转换成数组的数据类型
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64))
        # vstack 将两个数组垂直方向堆叠成一个新数组
        # from_numpy是将numpy中的ndarray转化成pytorch中的tensor
        # indices是coo的索引
    values = torch.from_numpy(sparse_mx.data)
        # values是coo的值
    shape = torch.Size(sparse_mx.shape)
        # coo的形状
    return torch.sparse.FloatTensor(indices,values,shape)
        # sparse.FloatTensor构造稀疏张量


def load_cora_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def accuracy_cora(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)


def get_degree_matrix(adj):

	return torch.diag(sum(adj))

def normalize_adj(adj):
	# Normalize adjacancy matrix according to reparam trick in GCN paper
	A_tilde = adj + torch.eye(adj.shape[0])
	D_tilde = get_degree_matrix(A_tilde)
	# Raise to power -1/2, set all infs to 0s
	D_tilde_exp = D_tilde ** (-1 / 2)
	D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

	# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
	norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
	return norm_adj

def normalize_adj_cuda(adj):
    if adj.device.index == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif adj.device.index == 1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif adj.device.index == 2:
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
	# Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0]).to(device)
    D_tilde = get_degree_matrix(A_tilde)
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

    return norm_adj

def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])     # k_hop_subgraph=子图的节点集(包括目标节点)、子图的边集、用来查询的节点集（中心节点集）、指示原始图g中的边是否在子图中的布尔数组Get all nodes involved  edge_index一般都是两行的形式，第一行edge_index[0]是边的起点，第二行edge_index[1]终点
	# k_hop_subgraph返回4个元组，分别是子图的节点集，子图的边集，用来查询的节点集（中心节点集），只是原始图的边是否在子图中的false和true数组
	neighbour_idx = edge_subset[0][~np.isin(edge_subset[0],node_idx)]  #得到目标节点的邻居节点原始index，并剔除目标节点
	edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
	# subgraph提取子图的方法
	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
	sub_feat = features[edge_subset[0], :]
	sub_labels = labels[edge_subset[0]]
	new_index = np.array([i for i in range(len(edge_subset[0]))])
	node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # 原始预测节点的index映射到新的index 格式={xx:xx}
	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
	return sub_adj, sub_feat, sub_labels, node_dict, neighbour_idx#neighbour_idx是原始index

#当adj不是对阵矩阵时，去n跳邻居
def get_neighbourhood_directed(node_idx, edge_index, n_hops, features, labels):
    edge_index_undirectd = to_undirected(edge_index[0])
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index_undirectd)
    # k_hop_subgraph返回4个元组，分别是子图的节点集，子图的边集，用来查询的节点集（中心节点集），只是原始图的边是否在子图中的false和true数组
    neighbour_idx = edge_subset[0][~np.isin(edge_subset[0].cpu(),node_idx)]  #得到目标节点的邻居节点原始index，并剔除目标节点
    edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
    # subgraph提取子图的方法
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].cpu().numpy(), new_index))        # 原始预测节点的index映射到新的index 格式={xx:xx}
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict, neighbour_idx#neighbour_idx是原始index
def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def safe_open(path, w):
	''' Open "path" for writing, creating any parent directories as needed.'''
	mkdir_p(os.path.dirname(path))
	return open(path, w)

def edges_unique(list): #(292,0),(0,292)去掉这种重复的边
    r = set()
    for e in list:
        if (e[1],e[0]) in r :
            continue
        r.add(tuple(e))
    return r


def edges_unique_list(list): #{(292,0),(362,0),(3,113)}和{(362,0), (292,0), (3,113)}去掉这种重复的边的集合
    r = set()
    for e in list:
        if (e[1],e[0]) in r :
            continue
        r.add(tuple(e))
    return r

# node_dict 新旧节点index相互转换 (327,330), 新index 变成旧index(415,418)
def new_idx_edge_into_old_idx(set_new, node_dict): #tuple_new形式=(327,330)
    set_old = []
    for i in list(set_new):
        first = list(node_dict.keys())[list(node_dict.values()).index(i[0])]
        second = list(node_dict.keys())[list(node_dict.values()).index(i[1])]
        set_old.append((first, second))
    return set(set_old)

def old_idx_edge_into_new_idx(list_old, node_dict): #tuple_new形式=(327,330)
    set_new = []
    for i in list(list_old):
        first = node_dict[i[0]]
        second = node_dict[i[1]]
        set_new.append((first, second))
    return set(set_new)


def create_symm_matrix_from_vec(vector, n_rows):#将向量vector 变成对称矩阵
	matrix = torch.zeros(n_rows, n_rows)
	idx = torch.tril_indices(n_rows, n_rows)  #返回矩阵下三角的idx

	matrix[idx[0], idx[1]] = vector
	symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t() #tril返回一个矩阵主对角线以下的下三角矩阵，其余用0补充;-1表示对角线不包含进去，也补充为0
	return symm_matrix


def create_symm_matrix_from_vec_cuda(vector, n_rows,device):#将向量vector 变成对称矩阵
	matrix = torch.zeros(n_rows, n_rows)
	idx = torch.tril_indices(n_rows, n_rows)  #返回矩阵下三角的idx
	if torch.cuda.is_available():
		matrix = matrix.to(device)
		idx = idx.to(device)
	matrix[idx[0], idx[1]] = vector
	symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t() #tril返回一个矩阵主对角线以下的下三角矩阵，其余用0补充;-1表示对角线不包含进去，也补充为0
	return symm_matrix



def encode(adj):
    encoded = adj.reshape(-1, adj.shape[0] ** 2)
    encoded = encoded.numpy().astype('int16')
    encoded = np.packbits(encoded, axis=1)
    return encoded


def decode(num_node, encoded, dtype=np.float32):
    decoded = np.unpackbits(encoded, axis=-1,count=num_node ** 2,)
    decoded = decoded.reshape(*encoded.shape[:-1], num_node,num_node)
    return decoded.astype(dtype)


def edges_motif_acc(perturb_edges_new, node_dict, y_pred_orig):
    perturb_edges_old = new_idx_edge_into_old_idx(perturb_edges_new, node_dict)
    edges_in_motif_count = 0
    perturb_edges_pred, edges_in_motif = [], []
    for j in perturb_edges_old:
        perturb_edges_pred.append((y_pred_orig[j[0]].item(), y_pred_orig[j[1]].item()))
        if y_pred_orig[j[0]] and y_pred_orig[j[1]]:
            edges_in_motif.append((j[0], j[1]))
            edges_in_motif_count += 1

    prop_correct = edges_in_motif_count / len(perturb_edges_old)

    return prop_correct

def add_edges_random(adj,num_edges):
    inv_adj = 1-adj
    edges = np.nonzero(inv_adj.numpy())
    edges_l = list(zip(edges[0], edges[1]))
    edges_idx = np.random.choice(range(len(edges_l)),size=num_edges)
    for idx in edges_idx:
        (row, col) = edges_l[idx]
        adj[row, col] = 1
        adj[col, row] = 1
    adj = adj*(torch.ones(size=adj.shape) - torch.eye(adj.shape[0],adj.shape[1])) #去掉对角线的1
    return adj



def from_adj_to_get_old_nodes_and_edges_idx(node_idx,n_hops,edge_index,features, labels):
	neighbor_node_idx1 = get_neighbourhood(node_idx, edge_index, n_hops, features, labels)[4]  # 一阶邻居
	sub_adj, _, _, node_dict, neighbor_node_idx2 = get_neighbourhood(node_idx, edge_index, n_hops, features, labels)  # 二阶邻居
	edges_involved = np.nonzero(sub_adj.numpy())
	edges_involved_unique = edges_unique(list(zip(edges_involved[0], edges_involved[1])))  # 因为是对称矩阵，所以去重
	#得到有边的元组，是原来的节点index
	edges_involved_old_idx = new_idx_edge_into_old_idx(edges_involved_unique, node_dict)
	# 得到原来的节点index
	node_involved_old = np.unique(np.concatenate((edges_involved[0], edges_involved[1]), axis=0))
	nodes_involved_old_idx = ([list(node_dict.keys())[list(node_dict.values()).index(j)] for j in node_involved_old])
	return nodes_involved_old_idx,edges_involved_old_idx


def contains_set(set1, set2):
    if (not len(set2 - set1)) and (not len(set1 - set2)):
        print('The two cf are not identical') #两个反事实不完全相同
        set_output1 = set1
        set_output2 = []
    elif (set1.issubset(set2)) or (set1.issuperset(set2)):
        print('The two cf are inclusion relations')
        if len(set1) < len(set2): #输出小子图
            set_output1 = set1
            set_output2 = []
        elif len(set1) > len(set2):
            set_output1 = []
            set_output2 = set2
    elif (not (set1.issubset(set2))) and (not (set1.issuperset(set2))):
        set_output1 = set1
        set_output2 = set2

    return set_output1, set_output2