
import torch
from torch import nn
import torch.nn.functional as F
from cfgflownet.nets.attention import LinearMultiHeadAttention


class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, widening_factor=4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.widening_factor = widening_factor

        self.lin2 = nn.Linear(input_size, widening_factor * output_size)
        self.gelu = nn.GELU()
        self.lin3 = nn.Linear(widening_factor * output_size, output_size)

    def forward(self, inputs):
        hiddens = self.lin2(inputs)
        hiddens = self.gelu(hiddens)
        hiddens = self.lin3(hiddens)
        # hiddens = hiddens.detach()
        return hiddens


class TransformerBlock(nn.Module):
    def __init__(self, n_node, sub_adj_vec_size, hidden_size, num_heads, key_size, widening_factor):
        super(TransformerBlock, self).__init__()
        self.key_size = key_size
        self.num_heads = num_heads
        self.widening_factor = widening_factor
        # self.dropout = dropout

        self.edge_embedding1 = nn.Linear(1, hidden_size) #input 由矩阵拉直了，所以size的平方
        self.norm1 = nn.LayerNorm(hidden_size+n_node)  # normalized_shape是输入tensor的最后的size，如输入数据的shape是(3, 4)，则normalized_shape=4
        # self.atten = nn.MultiheadAttention(256, num_heads)
        self.atten = LinearMultiHeadAttention(n_node, hidden_size, num_heads, key_size)

        self.lin1 = nn.Linear(hidden_size, n_node)

        self.edge_embedding2 = nn.Linear(1, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size + n_node)

        self.dense = DenseBlock(input_size=hidden_size + n_node, output_size=n_node,widening_factor=self.widening_factor)




    def forward(self, feat_embedding, sub_adj_vec):
        edge_embedding = self.edge_embedding1(sub_adj_vec) #inputs:36864*1 ->  花费gpu100
        h_norm = self.norm1(torch.cat((edge_embedding, feat_embedding.T),dim=1)) #h_norm:18528*448 花费gpu546
        # h_norm = F.dropout(h_norm, self.dropout, training=self.training)
        h_atten = self.atten(h_norm, h_norm, h_norm)#h_atten:192*256 花费gpu1638
        h_atten = self.lin1(h_atten)
        hiddens = feat_embedding.T + h_atten   #

        edge_embedding = self.edge_embedding2(sub_adj_vec)
        h_norm = self.norm2(torch.cat((edge_embedding, hiddens),dim=-1))#h_norm:192*512 花费gpu546
        h_dense = self.dense(h_norm) #花费gpu900
        hiddens = hiddens + h_dense

        return hiddens










