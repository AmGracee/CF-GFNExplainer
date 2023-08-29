
import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy

import torch
from gym.spaces import Dict, Box, Discrete
import torch.nn.functional as F
from cfgflownet.scores.base import Scorer,score_fn
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

    def step(self, node_idx, new_idx, sub_feat, sub_adj, sub_pred_orig, gcn_model, actions, num_edges_orig, logs):
        num_nodes = sub_adj.shape[0]
        sources = torch.div(actions,num_nodes,rounding_mode='floor')
        targets = torch.fmod(actions, num_nodes)
        dones = (sources == num_nodes) #dones denote stop
        # num_samples = torch.sum(~dones)
        sources, targets = sources[~dones], targets[~dones]

        # Record terminal state sf
        terminal_state = {}
        logs['is_exploration'] = logs['is_exploration'].type(torch.bool).squeeze(1)
        same_idx = [i for i,v in enumerate(dones) if v == ~logs['is_exploration'][i] and v == True]
        if torch.any(dones & ~logs['is_exploration']):
            terminal_state = {'node_idx': node_idx,
                              'new_idx': new_idx,
                              'terminal_state': self.encode(self._state['sub_adj'][dones & ~logs['is_exploration']]),
                              'cf_indicator': self._state['cf_indicator'][same_idx],
                              'order':self.encode(self._state['order'][dones & ~logs['is_exploration']])}

        if not torch.all(self._state['mask'][~dones, sources, targets]):
            raise ValueError('Some actions are invalid: the edge to be removed isn’t already in the sub_adj.')


        # Update the adjacency matrices
        self._state['sub_adj'][~dones, sources, targets] = 0
        self._state['sub_adj'][~dones, targets, sources] = 0
        self._state['sub_adj'][dones] = sub_adj.clone() # state is S_0 when action is a stop action.

        # Update the masks matrices
        self._state['mask'] = self._state['sub_adj'] #mask=8*6*6
        # sum(sum(self._state['mask'][0]))

        # Update the order, the sequence of removing edges
        self._state['order'][~dones, sources, targets] = self._state['num_edges'][~dones]
        self._state['order'][~dones, targets, sources] = self._state['num_edges'][~dones]
        self._state['order'][dones] = -1

        # Update the number of edges
        self._state['num_edges'][~dones] -= 1
        # print(self._state['num_edges'])
        num_edges_orig = num_edges_orig.clone().type(torch.int)
        self._state['num_edges'][dones] = num_edges_orig # num_edges is num_edges_orig when action is a stop action.

        # Compute reward scores: delta-scores log R(G_prime) - log R(G)
        reward, output, pred, output_change = score_fn(new_idx, self._state['sub_adj'], sub_feat, gcn_model, sub_pred_orig, self.sub_output_orig) #self._state['sub_adj']=移了边的adj

        self._state['pred'] = pred

        self._state['score'] = reward

        self._state['cf_indicator'] = (self._state['pred'] != sub_pred_orig)
        # print(self._state['cf_indicator'])
        cf_states,cf_example_min,  = [], []
        perturb_edges = []
        edges_acc = []
        if torch.any(self._state['cf_indicator']):
            cf_states = [node_idx,
                         new_idx,
                         sub_adj,
                         self._state['sub_adj'][self._state['cf_indicator']].cpu().numpy(),  #cf_adj
                         sub_pred_orig.item(),
                         [i.item() for i in self._state['pred'][self._state['cf_indicator']]],  # cf_pred
                         [i.item() for i in num_edges_orig - self._state['num_edges'][self._state['cf_indicator']]],  # num_removed_edges
                         output_change[self._state['cf_indicator']],  # fidelity-prob
                         self._state['order'][self._state['cf_indicator']] #order
                         ]
            for i in range(len(cf_states[3])):
                perturb_edges.append(edges_unique(np.nonzero(sub_adj.cpu() - cf_states[3][i]).tolist()))

            cf_states.append(perturb_edges)

        return (deepcopy(self._state), reward, dones, cf_states, terminal_state)


    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.sub_adj.shape[1] ** 2)
        encoded = encoded.cpu().numpy().astype('int16')
        encoded = np.packbits(encoded, axis=1)
        return encoded












