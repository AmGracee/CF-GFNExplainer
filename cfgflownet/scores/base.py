


import torch
import torch.nn.functional as F
from src.utils.utils import normalize_adj_cuda





# 计分器 trained gcn：过了gcn的output 先log再exp
def Scorer(node_idx, sub_adj, sub_features, gcn_model):
    sub_adj = sub_adj.cpu()
    R_score_batch, pred_batch = torch.tensor([]), torch.tensor([])
    for i in range(len(sub_adj)):
        norm_adj = normalize_adj_cuda(sub_adj[i])
        output = gcn_model(sub_features, norm_adj)
        pred = torch.argmax(output, dim=1)[node_idx].unsqueeze(0)
        R_score = output[node_idx][pred]
        R_score_batch = torch.cat((R_score_batch,R_score))
        pred_batch = torch.cat((pred_batch, pred))

    return R_score_batch, pred_batch


def score_fn(node_idx, sub_adj, sub_feat, gcn_model, sub_pred_orig, sub_output_orig):

    reward_batch,output_batch,pred_batch,output_change_batch = [],[],[],[]
    for i in range(len(sub_adj)):
        norm_adj = normalize_adj_cuda(sub_adj[i])
        assert not gcn_model.training
        output = gcn_model(sub_feat, norm_adj).detach()
        output_batch.append(output[node_idx])
        output_change = sub_output_orig - torch.softmax(output[node_idx],dim=0)[sub_pred_orig]
        output_change_batch.append(output_change)
        pred = torch.argmax(output, dim=1)[node_idx].unsqueeze(0)
        pred_batch.append(pred)
        reward = torch.exp(F.cross_entropy(torch.softmax(output[node_idx],dim=0), sub_pred_orig, reduction='none')).unsqueeze(0).detach()
        reward_batch.append(reward)

    return torch.stack(reward_batch,dim=1).squeeze(0), torch.stack(output_batch,dim=0), torch.stack(pred_batch,dim=1).squeeze(0),torch.stack(output_change_batch,dim=0).squeeze(0)

def score_fn_graph_classification(sub_adj, sub_feat, gcn_model, sub_pred_orig, sub_output_orig):

    reward_batch,output_batch,pred_batch,output_change_batch = [],[],[],[]
    for i in range(len(sub_adj)):
        norm_adj = normalize_adj_cuda(sub_adj[i])
        assert not gcn_model.training
        output = gcn_model(sub_feat, norm_adj).detach().squeeze()
        output_batch.append(output)
        output_change = sub_output_orig - torch.softmax(output,dim=0)[sub_pred_orig]
        output_change_batch.append(output_change)
        pred = torch.argmax(output, dim=0).unsqueeze(0)
        pred_batch.append(pred)
        reward = torch.exp(F.cross_entropy(torch.softmax(output,dim=0), sub_pred_orig, reduction='none')).unsqueeze(0).detach()
        reward_batch.append(reward)

    return torch.stack(reward_batch,dim=1).squeeze(0), torch.stack(output_batch,dim=0), torch.stack(pred_batch,dim=1).squeeze(0),torch.stack(output_change_batch,dim=0).squeeze(0)