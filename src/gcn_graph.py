import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn.parameter import Parameter
import torch.nn as nn


# gcn.layer
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_graph(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_hidden,
            num_output,
            device,
            dropout=0.1
    ):
        super(GCN_graph, self).__init__()

        self.device = device

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.conv1 = GraphConvolution(num_input, num_hidden)
        self.conv2 = GraphConvolution(num_hidden, num_hidden)
        self.conv3 = GraphConvolution(num_hidden, num_hidden)

        self.lin1 = torch.nn.Linear(num_hidden*2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_output)

        self.dropout = dropout

    def forward(self, x, adj):

        batch = torch.zeros(x.shape[0]).long().to(self.device)

        x = F.relu(self.conv1(x, adj))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, adj))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, adj))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        node_embs = x

        x = x1 + x2 + x3

        graph_emb = x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))

        x = self.lin3(x)

        return F.log_softmax(x, dim=-1)
