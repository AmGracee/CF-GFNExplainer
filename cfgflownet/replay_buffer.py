import numpy as np
import math
import torch
from numpy.random import default_rng
from src.utils.utils import encode


class ReplayBuffer:
    def __init__(self, capacity, sub_adj):
        self.capacity = capacity
        self.sub_adj = sub_adj
        shape = sub_adj.shape
        # dtype = np.dtype([
        #     ('is_exploration', torch.cuda.BoolTensor, (1,)),
        #     ('sub_adj', torch.cuda.FloatTensor,(shape)),
        #     ('num_edges',torch.cuda.IntTensor, (1,)),
        #     ('actions', torch.cuda.IntTensor, (1,)),
        #     ('delta_scores', torch.cuda.LongTensor, (1,)),
        #     ('mask', torch.cuda.FloatTensor),
        #     ('next_adj', torch.cuda.FloatTensor),
        #     ('next_mask',torch.cuda.FloatTensor),
        #     ('scores', torch.cuda.LongTensor, (1,)),
        # ])
        self._replay = [dict([
                         ('cf_indicator', torch.zeros(0, dtype=torch.bool)),
                         ('next_cf_indicator', torch.zeros(0, dtype=torch.bool)),
                         ('is_exploration',torch.zeros(1)),
                         ('sub_adj', torch.zeros(shape)),
                         ('num_edges',torch.zeros(1)),
                         ('actions', torch.zeros(1)),
                         ('delta_scores', torch.zeros(1)),
                         ('mask', torch.zeros(shape)),
                         ('next_adj', torch.zeros(shape)),
                         ('next_mask',torch.zeros(shape)),
                         ('scores', torch.zeros(1))]) for i in range(capacity)]
        self._index = 0
        self._is_full = False
        self._prev = torch.full((capacity,), -1, dtype=torch.int64)

    @property
    def dummy(self):
        shape = (self.sub_adj.shape[0], self.sub_adj.shape[0])
        return {
            'sub_adj': self.sub_adj,
            'num_edges': np.zeros((1,), dtype=np.int_),
            'actions': np.zeros((1,), dtype=np.int_),
            'delta_scores': np.zeros((1,), dtype=np.float_),
            'mask': np.zeros(shape, dtype=np.float32),
            'next_adj': np.zeros(shape, dtype=np.float32),
            'next_mask': np.zeros(shape, dtype=np.float32)
        }

    def add(self,observations,actions,is_exploration,next_observations,delta_scores,dones,prev_indices=None):
        indices = torch.full((dones.shape[0],), -1)
        if torch.all(dones):
            return indices
        num_samples = torch.sum(~dones)  # 采样了7个action，因为7个action有效
        add_idx = torch.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)  # 查看缓存的内存capacity是否满了。|=位置或，与逻辑或结果一样，只是过程不一样。
        self._index = (self._index + num_samples) % self.capacity  # 查看缓存里当前的sample数，即缓存里面有多少action。
        indices[~dones] = add_idx  # 当~dones的值为true才把add_idx的值赋值给indices，如果~dones为false，则为默认值-1

        data = {
            'cf_indicator': observations['cf_indicator'][~dones],
            'sub_adj': self.encode(observations['sub_adj'][~dones]),  #采取了stop action之前的observations不放入
            'num_edges': observations['num_edges'][~dones],
            'actions': actions[~dones],
            'delta_scores': delta_scores[~dones], # next_adj的reward
            'mask': self.encode(observations['mask'][~dones]),
            'next_cf_indicator': next_observations['cf_indicator'][~dones],
            'next_adj': self.encode(next_observations['sub_adj'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones]),

            'is_exploration': is_exploration[~dones],
            'scores': observations['score'][~dones]} #sub_adj的reward

        #sub_adj next_adj推入缓存器
        for name in data:
            for i in add_idx:
                # print(i)
                self._replay[i][name] = data[name][i-add_idx[0]]

        if prev_indices is not None:
            self._prev[add_idx] = prev_indices[~dones]

        return indices

    def sample(self, batch_size, rng, device):
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = []

       #decode


        for i in indices:
            samples.append(self._replay[i])# 选256个sample
        sub_adj = torch.cat((torch.tensor(self.decode(samples[0]['sub_adj'])).unsqueeze(0),torch.tensor(self.decode(samples[1]['sub_adj'])).unsqueeze(0)),dim=0)
        mask = torch.cat((torch.tensor(self.decode(samples[0]['mask'])).unsqueeze(0),torch.tensor(self.decode(samples[1]['mask'])).unsqueeze(0)),dim=0)
        next_adj = torch.cat((torch.tensor(self.decode(samples[0]['next_adj'])).unsqueeze(0), torch.tensor(self.decode(samples[1]['next_adj'])).unsqueeze(0)), dim=0)
        next_mask = torch.cat((torch.tensor(self.decode(samples[0]['next_mask'])).unsqueeze(0), torch.tensor(self.decode(samples[1]['next_mask'])).unsqueeze(0)), dim=0)
        num_edges = torch.cat((samples[0]['num_edges'].unsqueeze(0),samples[1]['num_edges'].unsqueeze(0)),dim=0)
        delta_scores = torch.cat((samples[0]['delta_scores'].unsqueeze(0),samples[1]['delta_scores'].unsqueeze(0)),dim=0)
        actions = torch.cat((samples[0]['actions'].unsqueeze(0),samples[1]['actions'].unsqueeze(0)),dim=0)

        for i in range(2, len(samples)):
            sub_adj = torch.cat((sub_adj, torch.tensor(self.decode(samples[i]['sub_adj'])).unsqueeze(0)), dim=0)
            mask = torch.cat((mask, torch.tensor(self.decode(samples[i]['mask'])).unsqueeze(0)), dim=0)
            next_adj = torch.cat((next_adj, torch.tensor(self.decode(samples[i]['next_adj'])).unsqueeze(0)), dim=0)
            next_mask = torch.cat((next_mask, torch.tensor(self.decode(samples[i]['next_mask'])).unsqueeze(0)), dim=0)
            num_edges = torch.cat((num_edges, samples[i]['num_edges'].unsqueeze(0)), dim=0)
            delta_scores = torch.cat((delta_scores, samples[i]['delta_scores'].unsqueeze(0)), dim=0)
            actions = torch.cat((actions, samples[i]['actions'].unsqueeze(0)), dim=0)


        # Convert structured array into dictionary
        return {
            'sub_adj': sub_adj.to(device),
            'num_edges': num_edges,
            'actions': actions,
            'delta_scores': delta_scores.to(device),
            'mask': mask.to(device),
            'next_adj': next_adj.to(device),
            'next_mask': next_mask.to(device)
        }


    def __len__(self):
        return self.capacity if self._is_full else self._index

    @property
    def transitions(self):
        return self._replay[:len(self)]

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.sub_adj.shape[0] ** 2)
        encoded = encoded.cpu().numpy().astype('int16')
        encoded = np.packbits(encoded, axis=1)
        return encoded

    def decode(self, encoded,dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1,count=self.sub_adj.shape[0] ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.sub_adj.shape[0],self.sub_adj.shape[0])
        return decoded.astype(dtype)








