
import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy

import torch
from gym.spaces import Dict, Box, Discrete
import torch.nn.functional as F
from cfgflownet.scores.base import Scorer,score_fn_graph_classification
from src.utils.utils import edges_unique, edges_unique_list,edges_motif_acc


class GFlowNetEnv(gym.vector.VectorEnv):
    def __init__(
            self,
            sub_adj,
            # sub_feat,
            num_envs,
            reward_orig,
            sub_output_orig,
            # node_dict,
            # y_pred_orig,
            # scorer,
            # num_workers,
            # context=None,
            cache_max_size=10_000):

        # self.scorer = scorer
        # self.num_workers = num_workers

        self.sub_adj = sub_adj
        # self.sub_feat = sub_feat
        self.reward_orig = reward_orig
        self.sub_output_orig = sub_output_orig
        # self.node_dict = node_dict
        # self.y_pred_orig = y_pred_orig
        self._state = None

        # if num_workers > 0:
        #     ctx = get_context(context) #创建进程,并设置进程启动方式spawn
        #
        #     self.in_queue = ctx.Queue() #使用消息机制queue队列实现进程间的通信
        #     self.out_queue = ctx.Queue()
        #     self.error_queue = ctx.Queue()
        #
        #     self.processes = []
        #     for index in range(num_workers): #有4个进程
        #         process = ctx.Process(
        #             target=self.scorer.forward(),
        #             args=(index, self.in_queue, self.out_queue, self.error_queue),  # args是target的参数
        #             daemon=True
        #         )
        #         process.start()

        shape = (self.sub_adj.shape[0], self.sub_adj.shape[0])
        max_edges = int(self.sub_adj.sum()/2) #sub_adj中存在的边的数量，即还可以删除的数量
        observation_space = Dict({
            'sub_adj': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'mask': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'num_edges': Discrete(max_edges),
            # 'pred_orig': Box(low=0, high=6, shape=(), dtype=np.int_),
            'score': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float_),
            'pred': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float_),
            'order': Box(low=-1, high=max_edges, shape=shape, dtype=np.int_),
            'cf_indicator': Box(low=0, high=1, shape=(), dtype=np.bool_)#指示该sub_adj是否是cf
        })
        action_space = Discrete(self.sub_adj.shape[0] + 1) #adj里面能删除的边的数量
        super().__init__(num_envs, observation_space, action_space)

    def reset(self):
        shape = (self.num_envs, self.sub_adj.shape[0], self.sub_adj.shape[0])
        # closure_T = torch.eye(self.sub_adj.shape[0], dtype=np.bool_)
        # self._closure_T = np.tile(closure_T, (self.num_envs, 1, 1))  # 将一个closure_T扩展成8个
        num_edges = int(self.sub_adj.sum()/2)
        self.sub_adj = torch.tile(self.sub_adj,(self.num_envs, 1, 1)) #将sub_adj拓展成8个
        self.reward_orig = torch.tile(self.reward_orig,(self.num_envs,))
        self._state = {
            'sub_adj': self.sub_adj,
            'mask': self.sub_adj,
            'num_edges': num_edges * torch.ones(size=(self.num_envs,),dtype=torch.int),
            # 'pred_orig' : self.pred_orig,
            'score': self.reward_orig,
            'pred': torch.zeros((self.num_envs,), dtype=torch.float),
            'order': torch.full(shape, -1, dtype=torch.int),  # 显示添边的顺序
            'cf_indicator': torch.zeros(size=(self.num_envs,),dtype=torch.bool)
        }
        return deepcopy(self._state)

    def step(self, graph_idx, sub_feat, sub_adj, sub_pred_orig, gcn_model, actions, num_edges_orig, logs):
        # 判断action是否是stop action
        num_nodes = sub_adj.shape[0]
        sources = torch.div(actions,num_nodes,rounding_mode='floor')
        targets = torch.fmod(actions, num_nodes)
        dones = (sources == num_nodes) #dones表示停止信号 true为没扰动
        # num_samples = torch.sum(~dones)
        sources, targets = sources[~dones], targets[~dones]

        # 记录terminal state sf
        terminal_state = {}
        logs['is_exploration'] = logs['is_exploration'].type(torch.bool).squeeze(1)
        same_idx = [i for i,v in enumerate(dones) if v == ~logs['is_exploration'][i] and v == True]
        if torch.any(dones & ~logs['is_exploration']):
            terminal_state = {
                              'graph_idx':graph_idx,
                              'terminal_state': self.encode(self._state['sub_adj'][dones & ~logs['is_exploration']]),
                              'cf_indicator': self._state['cf_indicator'][same_idx],
                              'order':self.encode(self._state['order'][dones & ~logs['is_exploration']])}

        # 判断删除的边是否存在adj
        if not torch.all(self._state['mask'][~dones, sources, targets]):
            raise ValueError('Some actions are invalid: the edge to be removed isn’t already in the sub_adj.')


        # Update the adjacency matrices
        self._state['sub_adj'][~dones, sources, targets] = 0
        self._state['sub_adj'][~dones, targets, sources] = 0 #adj 是对称的
        self._state['sub_adj'][dones] = sub_adj.clone() #dones为true是stop action ，state置为初始值

        # Update the masks matrices
        self._state['mask'] = self._state['sub_adj'] #mask=8*6*6
        # sum(sum(self._state['mask'][0]))

        # Update the order
        self._state['order'][~dones, sources, targets] = self._state['num_edges'][~dones]
        self._state['order'][~dones, targets, sources] = self._state['num_edges'][~dones]
        self._state['order'][dones] = -1  #stop action全置为-1

        # Update the number of edges
        self._state['num_edges'][~dones] -= 1  # remove 1条边
        # print(self._state['num_edges'])
        num_edges_orig = num_edges_orig.clone().type(torch.int)
        self._state['num_edges'][dones] = num_edges_orig #stop action的边重置为原始边的数量

        # Get the difference of log-rewards: delta-scores log R(G_prime) - log R(G)
        reward, output, pred, output_change = score_fn_graph_classification(self._state['sub_adj'], sub_feat, gcn_model, sub_pred_orig, self.sub_output_orig) #self._state['sub_adj']=移了边的adj
        # score_orig = torch.tile(score_orig, (1, self.num_envs)).squeeze(0)
        # delta_scores = (torch.log(scores) - torch.log(score_orig)).cuda()
        # delta_scores = -F.nll_loss(output[0],sub_pred_orig)

        self._state['pred'] = pred

        # score 当前G与初始G0的分数差
        self._state['score'] = reward
        # self._state['score'][dones] = 0 #非终止state的reward=0
        # self._state['score'] = self._state['score'].detach() #注意 score是带梯度的requires_grad=true是不能进行deepcopy，要先detach，然后再torch.requires_grad = True就行。

        self._state['cf_indicator'] = (self._state['pred'] != sub_pred_orig)
        # print(self._state['cf_indicator'])
        cf_states,cf_example_min,  = [], []
        perturb_edges = []
        edges_acc = []
        if torch.any(self._state['cf_indicator']):
            cf_states = [
                         graph_idx,
                         self.sub_adj[0],  #sub_adj
                         self._state['sub_adj'][self._state['cf_indicator']].cpu().numpy(),  #cf_adj
                         sub_pred_orig.item(),
                         [i.item() for i in self._state['pred'][self._state['cf_indicator']]],  #pred
                         [i.item() for i in num_edges_orig - self._state['num_edges'][self._state['cf_indicator']]],  # num_removed_edges 删了多少条边,除以2
                         output_change[self._state['cf_indicator']],  # fidelity-prob
                         self._state['order'][self._state['cf_indicator']] #反事实的删边顺序
                         ]
            for i in range(len(cf_states[2])):#cf_states[2] is cf_adj
                perturb_edges.append(edges_unique(np.nonzero(sub_adj.cpu() - cf_states[2][i]).tolist()))
            # for j in perturb_edges:
            #     edges_acc.append(edges_motif_acc(j,self.node_dict,self.y_pred_orig))
            cf_states.append(perturb_edges)

        return (deepcopy(self._state), reward, dones, cf_states, terminal_state)


    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.sub_adj.shape[1] ** 2)
        encoded = encoded.cpu().numpy().astype('int16')
        encoded = np.packbits(encoded, axis=1)
        return encoded












