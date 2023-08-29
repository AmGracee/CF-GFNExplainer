import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from cfgflownet.nets.base import elu_feature_map

class LinearMultiHeadAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, n_node, hidden_size, num_heads, key_size, eps=1e-6):
        super(LinearMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = key_size
        self.eps = eps

        self.lin1 = nn.Linear(n_node + hidden_size, num_heads * key_size)
        self.lin2 = nn.Linear(num_heads * key_size, num_heads * key_size)

    def forward(self, queries, keys, values, attn_mask=None):
        # Apply the feature map to the queries and keys
        feature_map = lambda x: F.elu(x) + 1.

        query_heads = self.lin1(queries).reshape(-1,4,64)
        key_heads = self.lin1(keys).reshape(-1,4,64)
        value_heads = self.lin1(values).reshape(-1,4,64)

        # query_heads = linear_projection(queries, self.key_size, self.num_heads)
        # key_heads = linear_projection(keys, self.key_size, self.num_heads)
        # value_heads = linear_projection(values, self.value_size, self.num_heads)

        query_heads = feature_map(query_heads)
        key_heads = feature_map(key_heads)

        key_values = torch.einsum('...thd,...thk->...hkd', key_heads, value_heads)
        # Compute the normalizer
        normalizer = 1. / (torch.einsum('...thd,...hd->...th', query_heads, torch.sum(key_heads, dim=-3)) + self.eps)
        # Finally compute and return the new values
        attn = torch.einsum('...thd,...hkd,...th->...thk', query_heads, key_values, normalizer)

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = torch.reshape(attn, (*queries.shape[:-1], -1))
        # attn_vec = self.lin2(attn_vec)

        return attn_vec


def linear_projection(x, head_size, num_heads):
    inputsieze = x.shape[1]
    y = nn.Linear(inputsieze, num_heads * head_size)(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, num_heads, head_size))