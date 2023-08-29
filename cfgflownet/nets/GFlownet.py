

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn

from tqdm.auto import trange

from cfgflownet.gflownet_utils import mask_logits
from src.utils.utils import create_symm_matrix_from_vec_cuda

from torch import nn
from cfgflownet.nets.transformers import TransformerBlock
from cfgflownet.gflownet_utils import log_policy
from cfgflownet.nets.MLP import MLP

MASKED_VALUE = -1e5

class gflownet(nn.Module):
    def __init__(self, n_node, nfeat, device, hidden_size=256, nheads=4, key_size=64, widening_factior=2):
        super(gflownet, self).__init__()

        self.n_node = n_node
        self.nfeat = nfeat
        self.sub_adj_vec_size = int(int((self.n_node * self.n_node - self.n_node) / 2) + self.n_node)


        self.feat_embedding = nn.Linear(self.nfeat, self.sub_adj_vec_size)
        self.transformer = TransformerBlock(self.n_node, self.sub_adj_vec_size, hidden_size, nheads, key_size, widening_factior)

        # self.dropout = dropout
        self.device = device


        self.mlp1 = MLP(self.n_node, 128, 1)
        self.mlp2 = MLP(self.n_node, 128, 1)


    def forward(self, sub_adj, sub_feat, masks):
        feat_embedding = self.feat_embedding(sub_feat)
        l_index = torch.tril_indices(sub_adj[0].shape[0], sub_adj[0].shape[0]) # sub_adj is a symmetric matrix,
        logits_batch = []

        for i in range(sub_adj.shape[0]):
            sub_adj_vec = sub_adj[i][l_index[0], l_index[1]].unsqueeze(1) # take the lower triangular matrix of sub_adj matrix

            embeddings = self.transformer(feat_embedding, sub_adj_vec)

            logit = torch.squeeze(self.mlp1(embeddings), dim=-1)
            logit = create_symm_matrix_from_vec_cuda(logit,self.n_node,self.device).reshape(-1)

            # terminate the trajectory Probability
            stop = self.mlp2(torch.mean(embeddings, dim=-2))

            mask = masks[i].reshape(logit.shape)
            masksed_logit = mask_logits(logit, mask)

            can_continue = torch.any(mask, dim=-1, keepdim=True)
            logp_continue = (F.log_softmax(-stop) + F.log_softmax(masksed_logit, dim=-1))
            logp_stop = F.logsigmoid(stop)

            logp_continue = torch.where(can_continue, logp_continue, torch.tensor(MASKED_VALUE).to(self.device))

            logp_stop = logp_stop * can_continue

            logits = torch.cat((logp_continue, logp_stop), dim=-1).unsqueeze(0).detach()
            logits_batch.append(logits)

        return torch.stack(logits_batch).squeeze(1)





def logits_head(embeddings, sub_adj):
    num_layers = 5
    for i in range(num_layers):
        embeddings = TransformerBlock(
            inputsize=sub_adj.shape[1],
            edges_embsize=embeddings.shape[1],
            embedding_size=128,
            num_heads=4,
            key_size=64,
            widening_factor=2)(embeddings, sub_adj)

    logits = MLP(256, 128, 1)(embeddings)
    return np.squeeze(logits, axis=-1)


def stop_head(embeddings, sub_adj):
    num_layers = 5
    for i in range(num_layers):
        embeddings = TransformerBlock(
            inputsize=sub_adj.shape[1],
            edges_embsize=embeddings.shape[1],
            embedding_size=128,
            num_heads=4,
            key_size=64,
            widening_factor=2)(embeddings, sub_adj)

    mean = torch.mean(embeddings, dim=-2)  # Average over edges
    return MLP(256, 128, 1)(mean)








