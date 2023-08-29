import pickle
import random
import numpy as np
import scipy.sparse as sp
import torch

from torch.nn import functional as F
from src.utils.utils import load_cora_data, load_npz, create_symm_matrix_from_vec, add_edges_random, sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils import is_undirected,to_undirected,dense_to_sparse,contains_self_loops,contains_isolated_nodes,to_dense_adj
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data,DataLoader


def load_syn1():
    with open("../data/syn1.pickle", "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_train = torch.sort(idx_train, descending=False)[0]
    idx_test = torch.tensor(data["test_idx"])
    idx_test = torch.sort(idx_test, descending=False)[0]
    valid_nodes = torch.tensor([3, 118, 152, 165, 167, 172, 174, 185, 186, 206, 210, 212, 215, 216, 218, 219, 221, 222, 225, 227, 228, 238,241, 243, 250, 251, 256, 265, 266, 267, 271, 272, 273, 283, 284, 289, 291, 295, 297, 298, 302, 312, 317, 322,327, 332, 337, 342, 347, 352, 357, 362, 367, 377, 397, 417, 427, 432, 437, 441, 443, 444, 447, 456, 458, 459,467, 469, 472, 481, 483, 484, 486, 488, 489, 491, 492, 493, 506, 508, 509, 521, 523, 524, 531, 532, 534, 551,553, 554, 561, 563, 564, 576, 578, 579, 591, 593, 594, 603, 604, 611, 613, 614, 621, 623, 624, 626, 628, 629,636, 637, 638, 639, 641, 643, 647, 651, 653, 654, 656, 657, 658, 659, 662, 668, 676, 678, 679, 681, 683, 684,686, 688, 689, 691, 693, 694, 698, 699])
    # valid_nodes = torch.tensor([228, 238,241, 243, 250, 251, 256, 265, 266, 267, 271, 272, 273, 283, 284, 289, 291, 295, 297, 298, 302, 312, 317, 322,327, 332, 337, 342, 347, 352, 357, 362, 367, 377, 397, 417, 427, 432, 437, 441, 443, 444, 447, 456, 458, 459,467, 469, 472, 481, 483, 484, 486, 488, 489, 491, 492, 493, 506, 508, 509, 521, 523, 524, 531, 532, 534, 551,553, 554, 561, 563, 564, 576, 578, 579, 591, 593, 594, 603, 604, 611, 613, 614, 621, 623, 624, 626, 628, 629,636, 637, 638, 639, 641, 643, 647, 651, 653, 654, 656, 657, 658, 659, 662, 668, 676, 678, 679, 681, 683, 684,686, 688, 689, 691, 693, 694, 698, 699])
    return adj, features, labels, idx_train, idx_test, valid_nodes


def load_syn4_save_pickle(args):
    with open("../data/synthetic/{}.pickle".format(args.dataset[:4]), "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()
    # adj随机连一些边
    adj = add_edges_random(adj,300)

    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_train = torch.sort(idx_train, descending=False)[0]
    idx_test = torch.tensor(data["test_idx"])
    args.dropout = 0.0
    idx_test = torch.sort(idx_test, descending=False)[0]
    args.nhops = 4
    valid_nodes = idx_test
    data_dict = {'adj': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_test': idx_test}
    file = open('../data/synthetic/{}_addedge300.pickle'.format(args.dataset), 'wb')
    pickle.dump(data_dict, file)
    file.close()

    return adj, features, labels, idx_train, idx_test, valid_nodes

def load_syn4_addedges():
    f = open('../data/syn4_addedge300.pickle','rb')
    dataset = pickle.load(f)
    adj = dataset['adj']
    features = dataset['features']
    labels = dataset['labels']
    idx_train = dataset['idx_train']
    idx_test = dataset['idx_test']
    valid_nodes = idx_test

    return adj, features, labels, idx_train, idx_test, valid_nodes


def load_syn41(dataset):
    with open("../data/synthetic/{}.pickle".format(dataset[:4]), "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()
    # adj随机连一些边
    adj = add_edges_random(adj,300)

    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_train = torch.sort(idx_train, descending=False)[0]
    idx_test = torch.tensor(data["test_idx"])
    # args.dropout = 0.0
    idx_test = torch.sort(idx_test, descending=False)[0]
    # args.nhops = 4
    valid_nodes = idx_test

    return adj, features, labels, idx_train, idx_test, valid_nodes

def load_syn5_save_pickle(args):
    with open("../data/synthetic/{}.pickle".format(args.dataset[:4]), "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()
    # adj随机连一些边
    adj = add_edges_random(adj, 300)

    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_train = torch.sort(idx_train, descending=False)[0]
    idx_test = torch.tensor(data["test_idx"])
    args.dropout = 0.0
    idx_test = torch.sort(idx_test, descending=False)[0]
    args.nhops = 4
    valid_nodes = idx_test

    data_dict = {'adj': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_test': idx_test}
    file = open('../data/synthetic/{}_addedge300.pickle'.format(args.dataset), 'wb')
    pickle.dump(data_dict, file)
    file.close()

    return adj, features, labels, idx_train, idx_test, valid_nodes

def load_syn5_addedges():
    f = open('../data/syn5_addedge300.pickle','rb')
    dataset = pickle.load(f)
    adj = dataset['adj']
    features = dataset['features']
    labels = dataset['labels']
    idx_train = dataset['idx_train']
    idx_test = dataset['idx_test']

    valid_nodes = idx_test

    return adj, features, labels, idx_train, idx_test, valid_nodes


def load_syn51(dataset):
    with open("../data/synthetic/{}.pickle".format(dataset[:4]), "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()
    # adj随机连一些边
    adj = add_edges_random(adj, 300)

    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_train = torch.sort(idx_train, descending=False)[0]
    idx_test = torch.tensor(data["test_idx"])
    # args.dropout = 0.0
    idx_test = torch.sort(idx_test, descending=False)[0]
    # args.nhops = 4
    valid_nodes = idx_test

    return adj, features, labels, valid_nodes
def load_cora():
    adj, features, labels, idx_train, idx_val, idx_test = load_cora_data()
    adj = adj.to_dense()
    valid_nodes = torch.tensor([500, 503, 505, 509, 510, 511, 514, 517, 518, 521, 522, 524, 525, 526, 529, 531, 538, 542, 545, 547, 548, 550,555, 557, 558, 559, 560, 564, 568, 570, 574, 577, 579, 582, 584, 585, 587, 589, 590, 591, 596, 597, 600, 603,607, 609, 616, 617, 619, 623, 625, 626, 627, 628, 630, 631, 632, 634, 639, 648, 649, 652, 653, 657, 660, 661,662, 663, 667, 668, 669, 672, 673, 674, 675, 677, 680, 681, 682, 683, 684, 690, 692, 693, 694, 698, 701, 703,704, 705, 706, 707, 708, 710, 711, 712, 756, 760, 766, 775, 776, 780, 782, 783, 786, 787, 789, 796, 804, 805,806, 807, 810, 811, 815, 820, 821, 822, 823, 824, 825, 827, 831, 836, 838, 843, 844, 845, 846, 847, 848, 849,850, 851, 852, 853, 858, 859, 861, 862, 864, 867, 869, 871, 873, 876, 877, 878, 879, 880, 885, 887, 893, 898,899, 900, 901, 903, 906, 908, 909, 911, 912, 915, 926, 929, 931, 933, 943, 944, 945, 949, 951, 952, 953, 954,955, 956, 959, 960, 963, 964, 965, 969, 970, 971, 975, 981, 984, 985, 986, 990, 991, 994, 997, 999, 1006, 1009,1012, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1027, 1029, 1031, 1032, 1034, 1035, 1036,1038, 1040, 1041, 1049, 1053, 1055, 1059, 1061, 1071, 1073, 1074, 1075, 1077, 1078, 1079, 1081, 1082, 1083,1084, 1086, 1089, 1091, 1092, 1096, 1100, 1101, 1104, 1108, 1109, 1111, 1112, 1118, 1120, 1123, 1124, 1129,1131, 1133, 1134, 1135, 1137, 1141, 1142, 1143, 1146, 1152, 1154, 1155, 1158, 1160, 1161, 1164, 1165, 1169,1170, 1172, 1173, 1174, 1175, 1176, 1182, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1197, 1198, 1204,1208, 1215, 1220, 1221, 1222, 1223, 1226, 1227, 1228, 1230, 1234, 1235, 1236, 1237, 1238, 1247, 1256, 1258,1259, 1260, 1261, 1262, 1263, 1264, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1275, 1276, 1282, 1283, 1284,1287, 1288, 1289, 1290, 1292, 1293, 1295, 1296, 1297, 1299, 1301, 1304, 1306, 1308, 1309, 1310, 1311, 1312,1314, 1315, 1316, 1318, 1321, 1322, 1325, 1327, 1328, 1329, 1335, 1336, 1337, 1339, 1341, 1346, 1350, 1351,1352, 1353, 1354, 1355, 1356, 1357, 1359, 1360, 1361, 1366, 1373, 1374, 1375, 1385, 1386, 1389, 1391, 1393,1396, 1399, 1400, 1401, 1402, 1403, 1405, 1406, 1407, 1412, 1414, 1422, 1423, 1424, 1425, 1428, 1430, 1431,1432, 1433, 1436, 1437, 1442, 1445, 1446, 1449, 1452, 1453, 1460, 1461, 1462, 1463, 1466, 1468, 1471, 1472,1476, 1479, 1480, 1481, 1482, 1484, 1485, 1487, 1489, 1491])

    return adj,features,labels,idx_train, idx_test,valid_nodes

def load_citeseer_symm(args):
    adj, features, labels = load_npz(f'../data/{args.dataset}/{args.dataset}.npz')
    adj = torch.Tensor(adj.todense())  # Does not include self loops
    loop_nodes = torch.tensor([0, 57, 60, 67, 82, 96, 116, 139, 145, 174, 186, 205, 220, 253, 262, 303, 346, 375, 379, 393, 405, 425, 498, 522, 533, 580, 611, 624, 627, 631, 706, 718, 766, 951, 972, 973, 1002, 1035, 1044, 1087, 1135, 1224, 1227, 1228, 1231, 1236, 1278, 1297, 1305, 1336, 1419, 1433, 1451, 1453, 1463, 1487, 1503, 1507, 1514, 1516, 1533, 1564, 1670, 1676, 1687, 1711, 1720, 1724, 1757, 1790, 1822, 1853, 1904, 1907, 1940, 1993, 2001, 2007, 2008, 2037, 2057, 2118, 2123, 2186, 2215, 2222, 2276, 2287, 2309, 2337, 2348, 2390, 2391, 2421, 2467, 2468, 2471, 2508, 2570, 2626, 2628, 2643, 2714, 2738, 2748, 2753, 2778, 2784, 2827, 2874, 2898, 2915, 2972, 2976, 2978, 2998, 3024, 3052, 3065, 3157, 3245, 3274, 3276, 3294])
    for i in loop_nodes:
        adj[i][i] = 0
    edge_index = dense_to_sparse(adj)
    edge_index_undirectd = to_undirected(edge_index[0])
    adj = to_dense_adj(edge_index_undirectd).squeeze()
    features = torch.Tensor(features.todense())
    labels = torch.tensor(labels).long()
    split = np.load(f'../data/{args.dataset}/' + args.dataset + '_split.npy',allow_pickle=True).item()
    idx_train = torch.tensor(split['train'])
    idx_train = torch.sort(idx_train, descending=False)[0]
    idx_val = torch.tensor(split['val'])
    idx_val = torch.sort(idx_val, descending=False)[0]
    idx_test = torch.tensor(split['test'])
    idx_test = torch.sort(idx_test,descending=False)[0]
    separate_nodes = [522, 1670, 498, 1507, 533, 1419, 706]
    valid_nodes = idx_test[~np.isin(idx_test, separate_nodes)]
    args.nhops = 3
    args.dropout = 0.5

    return adj, features, labels, valid_nodes


def load_citeseer_symm_loop(args):
    adj, features, labels = load_npz(f'../data/{args.dataset}/{args.dataset}.npz')
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])  # 将非对称邻接矩阵转变为对称邻接矩阵（有向图转无向图）
    adj[adj > 1] = 1
    adj = torch.Tensor(adj.todense())
    features = torch.Tensor(features.todense())
    labels = torch.tensor(labels).long()
    split = np.load(f'../data/{args.dataset}/' + args.dataset + '_split.npy',allow_pickle=True).item()
    idx_train = torch.tensor(split['train'])
    idx_val = torch.tensor(split['val'])
    idx_test = torch.tensor(split['test'])
    idx_test = torch.sort(idx_test,descending=False)[0]
    # isolated_nodes = [522, 1670, 498, 1507, 533, 1419, 706]
    # valid_nodes = idx_test[~np.isin(idx_test, isolated_nodes)]
    args.nhops = 3
    args.dropout = 0.5

    return adj, features, labels, idx_test


def load_citeseer_symm_save_pickle(args):
    adj, features, labels = load_npz(f'../data/{args.dataset}/{args.dataset}.npz')
    adj = torch.Tensor(adj.todense())  # Does not include self loops
    full_nodes = list(range(0,adj.shape[0]))
    edge_index = dense_to_sparse(adj)
    '将有向图变成无向图，即将adj变成对称矩阵'
    edge_index_undirectd = to_undirected(edge_index[0])
    adj = to_dense_adj(edge_index_undirectd).squeeze()
    adj[adj > 1] = 1
    'adj去除自环'
    for i in np.arange(len(adj)):
        if adj[i][i]>0:
            adj[i][i] = 0
            # print(i)
    isolated_nodes = []

    for n_i in np.arange(len(adj)):
        if adj[n_i].sum() < 1 and adj[:][n_i].sum() < 1:
            # print(n_i)
            isolated_nodes.append(n_i)
    '去除孤立节点'
    adj_np = adj.numpy()

    adj_np = np.delete(adj_np,isolated_nodes,0)
    adj_np = np.delete(adj_np, isolated_nodes, 1)
    # print(adj_np.sum())
    adj = torch.tensor(adj_np)
    # edge_index = dense_to_sparse(adj)


    features = torch.Tensor(features.todense())
    features = features[list(set(full_nodes)-set(isolated_nodes))]
    labels = torch.tensor(labels).long()
    labels = labels[list(set(full_nodes) - set(isolated_nodes))]
    split = np.load(f'../data/{args.dataset}/' + args.dataset + '_split.npy',allow_pickle=True).item()
    idx_train = torch.tensor(split['train'])
    idx_train = torch.sort(idx_train, descending=False)[0]
    # idx_val = torch.tensor(split['val'])
    idx_test = torch.tensor(split['test'])
    idx_test = torch.sort(idx_test,descending=False)[0]
    idx_train = idx_train[~np.isin(idx_train, isolated_nodes)]
    # idx_val = idx_val[~np.isin(idx_val, isolated_nodes)]
    idx_test = idx_test[~np.isin(idx_test, isolated_nodes)]
    data_dict = {'adj':adj,'features':features,'labels':labels,'idx_train':idx_train,'idx_test':idx_test}
    file = open('../data/citeseer/citeseer_remove_isolated_node.pickle','wb')
    pickle.dump(data_dict,file)
    file.close()
    args.nhops = 3
    args.dropout = 0.5

    return adj, features, labels, idx_train, idx_test


def load_citeseer():
    f = open('../data/citeseer/citeseer_remove_isolated_node.pickle','rb')
    dataset = pickle.load(f)
    adj = dataset['adj']
    features = dataset['features']
    labels = dataset['labels']
    idx_train = dataset['idx_train']
    idx_test = dataset['idx_test']
    valid_nodes = torch.tensor([1,14,18,19,23,25,38,44,47,58,75,79,80,84,91,97,99,101,113,114,117,124,127,131,136,138,150,152,154,160,161,181,182,186,197,204,209,219,220,224,232,239,258,259,263,269,270,274,278,283,284,285,286,288,301,315,318,319,338,348,362,372,373,374,378,381,387,388,397,402,403,406,412,423,425,444,450,456,459,466,470,479,496,511,514,515,520,524,536,540,545,546,550,551,552,553,554,556,562,568,576,577,587,588,590,604,611,614,616,626,628,638,643,651,662,664,668,671,672,679,683,689,695,700,701,705,707,708,709,713,717,719,723,735,744,770,775,786,788,790,805,810,811,814,827,832,833,851,856,859,885,892,893,896,899,903,908,909,912,918,919,922,924,927,931,933,937,947,948,954,965,969,972,978,980,981,983,984,985,986,987,988,991,995,996,1001,1009,1011,1013,1036,1041,1044,1046,1053,1057,1059,1061,1075,1077,1078,1083,1093,1094,1095,1103,1108,1113,1114,1132,1137,1138,1156,1164,1167,1174,1176,1184,1189,1199,1215,1216,1223,1226,1229,1239,1241,1248,1249,1250,1254,1275,1283,1285,1293,1294,1295,1312,1330,1335,1346,1363,1364,1366,1368,1372,1373,1376,1378,1382,1386,1391,1393,1394,1401,1408,1411,1414,1417,1426,1439,1440,1446,1448,1449,1456,1457,1458,1468,1472,1476,1479,1480,1490,1499,1502,1509,1512,1519,1525,1526,1551,1569,1570,1577,1579,1583,1586,1589,1592,1594,1595,1597,1598,1614,1617,1650,1652,1657,1665,1667,1669,1690,1691,1696,1700,1704,1705,1710,1713,1720,1736,1744,1750,1751,1758,1764,1775,1776,1777,1778,1808,1814,1820,1821,1824,1832,1833,1839,1858,1862,1867,1871,1872,1880,1882,1883,1888,1892,1898,1922,1926,1944,1969,1970,1972,1974,1975,1982,1998,1999,2000,2009,2020,2022,2023,2030,2033,2037,2041,2046,2059,2065,2079,2080,2088,2095,2096,2097,2102,2104,2105])
    return adj, features, labels, idx_train, idx_test, valid_nodes

def load_mutag():
    f = open('../data/mutag.pickle', 'rb')
    dataset = pickle.load(f)
    adj = dataset['adj']
    features = dataset['features']
    labels = dataset['labels']
    graph_idx = dataset['graph_idx']
    idx_train = graph_idx[:150]
    idx_test = graph_idx[150:]
    valid_graph_idx = graph_idx

    return adj, features, labels, idx_train, idx_test, valid_graph_idx

def load_mutag_symm_save_pickle():
    dataset = TUDataset('../data/TUDataset', name='MUTAG')  # 加载数据集
    data_list = [dataset.get(idx) for idx in range(len(dataset))]
    random.shuffle(data_list)
    graph_idx,sub_adj,sub_feat,sub_feat,sub_label,num_node,num_edge = [],[],[],[],[],[],[]
    for i in range(len(data_list)):
        data = data_list[i]
        graph_idx.append(i)
        edge_index = data.edge_index
        sub_adj.append(to_dense_adj(edge_index).squeeze())
        num_node.append(to_dense_adj(edge_index).squeeze().shape[0])
        num_edge.append((to_dense_adj(edge_index).squeeze().sum().item()) / 2)
        sub_feat.append(data.x)
        sub_label.append(data.y.item())
    sub_label = torch.tensor(sub_label)
    idx_train = graph_idx[:150]
    idx_test = graph_idx[150:]
    data_dict = {'graph_idx': graph_idx, 'adj': sub_adj, 'features': sub_feat, 'labels': sub_label, 'num_node': num_node, 'num_edge': num_edge,'idx_train':idx_train,'idx_test':idx_test}
    file = open('../data/mutag/mutag.pickle', 'wb')
    pickle.dump(data_dict, file)
    file.close()

    return sub_adj, sub_feat, sub_label, idx_train,idx_test,graph_idx

def load_mutagenicity_symm_save_pickle():
    dataset = 'Mutagenicity'
    pri = f'../data/{dataset}/raw/{dataset}_'

    file_edges = pri + 'A.txt'
    file_edge_masks = pri + 'edge_gt.txt'
    file_graph_indicator = pri + 'graph_indicator.txt'
    file_graph_labels = pri + 'graph_labels.txt'
    file_node_labels = pri + 'node_labels.txt'

    edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)  # 加载边数据
    edge_masks = np.loadtxt(file_edge_masks, delimiter=',').astype(np.int32)  # 加载边mask
    graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)  # 加载每个节点的图编号
    graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)  # 加载图标签
    node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)  # 加载节点标签

    # 获取节点对应的图编号
    num_nodes = 131488
    starts = [1]  # 每个图的起始节点编号
    node2graph = {}  # 保存节点对应的图编号
    for node_index, graph_index in enumerate(graph_indicator):
        node2graph[node_index + 1] = graph_index - 1
        if node_index and graph_index != graph_indicator[node_index - 1]:
            starts.append(node_index + 1)

    dataset = []
    # 获取图对应的边，节点，节点属性，边mask，并保存在process路径当中
    edge_index = [[], []]
    edge_mask = []

    for cnt, [(st, ed), mask] in enumerate(zip(edges, edge_masks)):
        st_id, ed_id = node2graph[st], node2graph[ed]
        assert st_id == ed_id, f'edges connecting different graphs, error here, please check.' \
                               f'{st, ed} graph id {st_id, ed_id}'

        graph_id = st_id
        if len(dataset) != graph_id or cnt == len(edge_masks) - 1:
            # 保存原图，创建新图
            if cnt < len(edge_masks) - 1:
                nodes_index = [i for i in range(starts[len(dataset)], starts[graph_id])]
            else:
                nodes_index = [i for i in range(starts[graph_id], num_nodes)]
            nodes_label = [node_labels[i] for i in nodes_index]  # 获取节点标签
            x = [[0] * 14 for _ in nodes_index]
            for i, j in enumerate(nodes_index):  # 根据节点标签设计节点属性
                j = node_labels[j]
                x[i][j] = 1
            data = Data(x=torch.tensor(x, dtype=torch.float),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_mask=torch.tensor(edge_mask, dtype=torch.float),
                        nodes_label=torch.tensor(nodes_label, dtype=torch.long),
                        y=torch.tensor(graph_labels[graph_id], dtype=torch.long))
            dataset.append(data)
            edge_index = [[], []]
            edge_mask = []
        edge_index[0].append(st - starts[graph_id])
        edge_index[1].append(ed - starts[graph_id])
        edge_mask.append(mask)




    graph_idx_train, sub_adj_train, sub_feat_train, sub_feat_train, sub_label_train, num_node_train, num_edge_train = [], [], [], [], [], [], []
    graph_idx, sub_adj, sub_feat, sub_feat, sub_label, num_node, num_edge = [], [], [], [], [], [], []
    for data in train_dataset:
        # graph_idx.append(i)
        edge_index = data.edge_index
        sub_adj_train.append(to_dense_adj(edge_index).squeeze())
        num_node_train.append(to_dense_adj(edge_index).squeeze().shape[0])
        num_edge_train.append((to_dense_adj(edge_index).squeeze().sum().item()) / 2)
        sub_feat_train.append(data.x)
        sub_label_train.append(data.y.item())
    idx_train = graph_idx[:150]
    idx_test = graph_idx[150:]
    data_dict = {'graph_idx': graph_idx, 'adj': sub_adj, 'features': sub_feat, 'labels': sub_label, 'num_node': num_node, 'num_edge': num_edge,'idx_train':idx_train,'idx_test':idx_test}
    file = open('../data/mutag/mutag.pickle', 'wb')
    pickle.dump(data_dict, file)
    file.close()

    return sub_adj, sub_feat, sub_label, idx_train,idx_test,graph_idx

