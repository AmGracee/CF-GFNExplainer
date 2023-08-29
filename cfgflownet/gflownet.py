import time
import numpy as np
import torch
from cfgflownet.nets.GFlownet import gflownet
from cfgflownet.gflownet_utils import uniform_log_policy,batch_random_choice, detailed_balance_loss
from src.utils.utils import linear_schedule,edges_motif_acc,contains_set,decode

class GFlowNet:
    def __init__(self, gcn_model, sub_adj, sub_feat, sub_pred_orig, num_edges_orig,node_dict,y_pred_orig, mask, env, replay, observations, device, delta=1.):
        super(GFlowNet, self).__init__()
        self.gcn_model = gcn_model
        self.sub_adj = sub_adj
        self.sub_feat = sub_feat
        self.sub_pred_orig = sub_pred_orig
        self.num_edges_orig = num_edges_orig
        self.node_dict = node_dict
        self.y_pred_orig = y_pred_orig
        self.mask = mask
        self.delta = delta
        self.env = env
        self.replay = replay
        self.observations = observations
        self.device = device
        self.GFN_model = gflownet(sub_adj.shape[0], sub_feat.shape[1], device)


    def explain(self, node_idx, new_idx, prefill, num_iterations, optimizer, min_exploration, batch_size, indices, rng):
        self.node_idx = node_idx
        self.new_idx = new_idx
        self.num_iterations = num_iterations
        self.prefill = prefill
        self.optimizer = optimizer
        self.min_exploration = min_exploration
        self.batch_size = batch_size
        self.indices = indices
        self.rng = rng

        self.exploration_schedule = linear_schedule(
            init_value=np.array(0.),
            end_value=np.array(1. - self.min_exploration),
            transition_steps=num_iterations // 2,
            transition_begin=prefill)

        batch_terminal_state = []
        best_cf_examples,best_cf_example = [],[]
        diversity, diversity_new = [], []
        fidelity_prob = []
        best_num_edges = 10000
        best_output_change = -10000
        self.num_cf_examples = 0
        max_fidelity_prob = -10000
        start = time.time()
        for iteration in range(prefill+num_iterations):
            cf_states, terminal_state = self.train(iteration)
            if cf_states != []:
                num_edges_min = min(cf_states[6])# cf_states[6] is num_edges

                edges_min_idx = cf_states[6].index(num_edges_min)
                cf_example_min = [node_idx.item(),
                                  new_idx,
                                  self.encode(cf_states[3][edges_min_idx]), # cf_states[3] is cf_adj
                                  cf_states[4],# sub_pred_orig
                                  cf_states[5][edges_min_idx], # cf_pred
                                  num_edges_min,
                                  cf_states[8][edges_min_idx].numpy(),# order is the sequence of removing edges
                                  ]

                if cf_example_min[5] < best_num_edges:
                    best_cf_examples.append(cf_example_min)
                    best_num_edges = cf_example_min[5]
                    self.num_cf_examples += 1
                    # print("iteration:{}, {} 个 CF examples for node_idx = {}".format(iteration, self.num_cf_examples, self.node_idx))
                    # break
                if len(terminal_state) != 0:
                    batch_terminal_state.append(terminal_state)
                    print('iteration:{},  Found terminal_state for node_idx = {}'.format(iteration, self.node_idx))

                if max(cf_states[7]).item() > best_output_change: # cf_states[7] is output_change
                    best_output_change = max(cf_states[7]).item()
                    fidelity_prob.append(best_output_change)

                for i in range(len(cf_states[3])):# cf_states[3] is cf_adj
                    diversity.append([cf_states[9][i], cf_states[8][i].numpy()]) # cf_states[9] is perturb,cf_states[8] is order

        spent_time = (time.time()- start)/60 # time is minute.
        print("{} 个 CF examples for node_idx = {}".format(self.num_cf_examples, self.node_idx))

        # Exclude containment relationships counterfactuals in diversity.
        for i in range(len(diversity)):
            for j in range(i+1,len(diversity)):
                if diversity[i][0] == []:
                    break
                elif diversity[j][0] == []:
                    continue

                cf1, cf2 = contains_set(diversity[i][0], diversity[j][0]) #self.sub_adj is sub_adj_orig
                diversity[i][0], diversity[j][0] = cf1, cf2

        for i in range(len(diversity)):
            if diversity[i][0]!=[]:
                diversity_new.append([diversity[i][0], diversity[i][1]])

        if len(best_cf_examples) > 0 :
            best_cf_example = best_cf_examples[-1]

        if len(fidelity_prob) > 0:
            max_fidelity_prob = max(fidelity_prob)

        return best_cf_example, batch_terminal_state, max_fidelity_prob, diversity_new, spent_time

    def train(self, iteration):
        self.GFN_model.train()
        self.optimizer.zero_grad()
        epsilon = self.exploration_schedule(iteration)
        # print('epsilon',epsilon)
        actions, logs = self.sample_action(self.observations, epsilon)
        # print(actions)
        self.sub_adj = self.sub_adj.to(self.device)
        next_observations, rewards, dones, cf_states, terminal_state = self.env.step(self.node_idx, self.new_idx, self.sub_feat, self.sub_adj, self.sub_pred_orig, self.gcn_model, actions, self.num_edges_orig, logs)
        delta_scores = next_observations['score'] - self.observations['score']
        self.indices = self.replay.add(
            self.observations,
            actions,
            logs['is_exploration'],
            next_observations,
            delta_scores,
            dones,
            prev_indices=self.indices)
        self.observations = next_observations
        # print('iteration: {} for node_idx: {}'.format(iteration,self.node_idx))

        if iteration >= self.prefill:
            # Update the parameters of the GFlowNet
            samples = self.replay.sample(self.batch_size, self.rng,self.device)  # 采样一千次action后，每次8个action采样。例如：第1-7次迭代后 某个adj增加7条边，第8次迭代采样的stopaction 所有adj清零。
            loss, _ = self.loss(samples)
            loss.backward()
            self.optimizer.step()
            print(
                'iteration: {}'.format(iteration),
                'loss: {:.4f}'.format(loss),
                '{} 个 CF examples for node_idx={}'.format(self.num_cf_examples,self.node_idx))
            # print('===================================================')

        return cf_states, terminal_state

    def loss(self, samples):
        log_pi_t = self.GFN_model(samples['sub_adj'], self.sub_feat, samples['mask'])
        log_pi_tp1 = self.GFN_model(samples['next_adj'], self.sub_feat, samples['next_mask'])  # 输出P(G'|G)

        log_pi_t.requires_grad,log_pi_tp1.requires_grad = True, True

        loss, logs = detailed_balance_loss(log_pi_t,
                                           log_pi_tp1,
                                           samples['actions'],
                                           samples['delta_scores'],
                                           samples['num_edges'],
                                           device=self.device,
                                           delta=self.delta,
                                          )
        return loss, logs


    def sample_action(self, observations, epsilon):
        with torch.no_grad():
            masks = observations['mask'] # size = 8*6*6
            # sub_feat = observations['sub_feat']
            sub_adj = observations['sub_adj'] # size = 8*6*6
            self.sub_feat = self.sub_feat.to(self.device)
            batch_size = sub_adj.shape[0]
            num_node = sub_adj.shape[1]
            epsilon = epsilon * torch.ones([batch_size, 1])
            log_pi = torch.zeros([batch_size,num_node*num_node+1]).to(self.device)
            is_exploration = torch.bernoulli(input=1 - epsilon).to(self.device)
            log_gflownet,log_uniform = torch.tensor([]), torch.tensor([])

            # Get the GFlowNet policy
            pi_exploration = (1-is_exploration).reshape(-1).bool()
            uniform_exploration = is_exploration.reshape(-1).bool()
            if torch.any(pi_exploration):
                log_gflownet = self.GFN_model(sub_adj[pi_exploration], self.sub_feat, masks[pi_exploration])
                log_pi[pi_exploration] = log_gflownet

            # Get uniform policy
            if torch.any(uniform_exploration):
                log_uniform = uniform_log_policy(masks[uniform_exploration])
                log_pi[uniform_exploration] = log_uniform

            # Sample actions
            actions = batch_random_choice(torch.exp(log_pi), masks, self.device)
            is_exploration = is_exploration.clone().int()
            # print(is_exploration.reshape(-1))
            logs = {'is_exploration': is_exploration}
        return (actions, logs)

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.sub_adj.shape[0] ** 2)
        encoded = encoded.astype('int16')
        encoded = np.packbits(encoded, axis=1)
        return encoded


