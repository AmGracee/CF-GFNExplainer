import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import trange
from numpy import random



def detailed_balance_loss(
        log_pi_t,
        log_pi_tp1,
        actions,
        delta_scores,
        num_edges,
        device,
        delta=1.
    ):
    r"""

    In practice, to avoid gradient explosion, we use the Huber loss instead of the L2-loss
    Parameters
    ----------
    log_pi_t :
        The log-probabilities $\log P_{\theta}(s' \mid s_{t})$, for all the next states $s'$, including the terminal
        state $s_{f}$.

    log_pi_tp1 :
        The log-probabilities $\log P_{\theta}(s' \mid s_{t+1})$, for all the next states $s'$, including the terminal
        state $s_{f}$.

    delta_scores :
        The delta-scores between state $s_{t}$ and state $s_{t+1}$, given by $\log R(s_{t+1}) - \log R(s_{t})$.
    num_edges :
        The number of edges in $s_{t}$.

    delta : float (default: 1.)
        The value of delta for the Huber loss.

    Returns
    -------
    loss :
        The detailed balance loss averaged over a batch of samples.

    logs : dict
        Additional information for logging purposes.
    """
    # Compute the forward log-probabilities
    log_pF = torch.take_along_dim(log_pi_t, actions.unsqueeze(1), dim=-1).squeeze(1)  # take_along_axis:log_pi_t按照action中的位置取值
    # log_pi_tp1 = P(sf|st+1) log_pi_t = P(sf|st)
    # Compute the backward log-probabilities

    log_pB = -torch.log1p(num_edges).to(device)
    # error = log_pF + log_pi_tp1[:, -1].detach() - log_pi_t[:, -1] - log_pB - delta_scores
    error = (torch.squeeze(delta_scores + log_pB - log_pF, dim=-1) + log_pi_t[:, -1] - log_pi_tp1[:, -1].detach())
    # loss = torch.mean(error)
    loss = torch.mean(F.huber_loss(error, torch.zeros_like(error), delta=delta))

    logs = {
        'error': error,
        'loss': loss,
    }
    return loss, logs


def log_policy(logits, stop, masks):
    masks = masks.reshape(logits.shape)
    masked_logits = mask_logits(logits, masks)
    can_continue = torch.any(masks, dim=-1, keepdim=True) #看mask中是否有1(是否还能添边)，任意一个元素为True，输出为True 相当于或

    logp_continue = (F.logsigmoid(-stop) + F.log_softmax(masked_logits, dim=-1))
    logp_stop = F.logsigmoid(stop)

    # In case there is no valid action other than stop
    if can_continue:
        logp_continue = logp_continue
    else:
        logp_continue = torch.ones_like(logp_continue) * MASKED_VALUE

    logp_stop = logp_stop * can_continue

    return torch.cat((logp_continue, logp_stop), dim=-1)



def posterior_estimate(
        gflownet,
        params,
        env,
        key,
        num_samples=1000,
        verbose=True,
        **kwargs
    ):
    """Get the posterior estimate of DAG-GFlowNet as a collection of graphs
    sampled from the GFlowNet.

    Parameters
    ----------
    gflownet : `DAGGFlowNet` instance
        Instance of a DAG-GFlowNet.

    params : dict
        Parameters of the neural network for DAG-GFlowNet. This must be a dict
        that can be accepted by the Haiku model in the `DAGGFlowNet` instance.

    env : `GFlowNetDAGEnv` instance
        Instance of the environment.

    key : jax.random.PRNGKey
        Random key for sampling from DAG-GFlowNet.

    num_samples : int (default: 1000)
        The number of samples in the posterior approximation.

    verbose : bool
        If True, display a progress bar for the sampling process.

    Returns
    -------
    posterior : np.ndarray instance
        The posterior approximation, given as a collection of adjacency matrices
        from graphs sampled with the posterior approximation. This array has
        size `(B, N, N)`, where `B` is the number of sample graphs in the
        posterior approximation, and `N` is the number of variables in a graph.

    logs : dict
        Additional information for logging purposes.
    """
    samples = []
    observations = env.reset()
    with trange(num_samples, disable=(not verbose), **kwargs) as pbar:
        while len(samples) < num_samples:
            order = observations['order']#order为添边的顺序
            actions, key, _ = gflownet.act(params, key, observations, 1.) #进行action采样
            observations, _, dones, _ = env.step(np.asarray(actions))  #dones中的true表示stop action

            samples.extend([order[i] for i, done in enumerate(dones) if done]) #当dones中有stop action就把当前的order放入sample中。
            pbar.update(min(num_samples - pbar.n, np.sum(dones).item()))
    orders = np.stack(samples[:num_samples], axis=0)  #有1000个添边的order
    logs = {
        'orders': orders,
    }
    return ((orders >= 0).astype(np.int_), logs)  #将order里的顺序都变成1

MASKED_VALUE = -1e5
def mask_logits(logits, masks):
    return masks * logits + (1. - masks) * MASKED_VALUE  #将mask中=0的概率赋值成-10000

def uniform_log_policy(masks):
    masks = masks.reshape(masks.shape[0], -1)#把mask维度变了
    num_edges = torch.sum(masks, dim=-1, keepdim=True) #把每一个mask都相加，则为允许添边的数量（因为mask=1表示允许连边）

    logp_stop = -torch.log1p(num_edges) #jnp.log1p=log(x+1)取对数，边越少，stop的p越大符合直觉
    logp_continue = mask_logits(logp_stop, masks)

    return torch.cat((logp_continue, logp_stop), dim=-1)


def batch_random_choice(probas, masks, device):
    # Sample from the distribution
    uniform = torch.zeros(size=(probas.shape[0], 1)).to(device)
    uniform = nn.init.uniform_(uniform) # uniform= 随机中产生一个数，设置key（key相当于随机种子）之后，得到的随机数运行都是一样的。
    cum_probas = torch.cumsum(probas, dim=1)  # cumsum对每一列求累加和
    samples = torch.sum(cum_probas < uniform, dim=1, keepdim=True)  #从第0条开始选，知道累加到大于uniform，就选那个条
    # In rare cases, the sampled actions may be invalid, despite having
    # probability 0. In those cases, we select the stop action by default.
    stop_mask = torch.ones((masks.shape[0], 1), dtype=masks.dtype).to(device) # Stop action is always valid
    masks = masks.reshape(masks.shape[0], -1)
    masks = torch.cat((masks, stop_mask), dim=1)  # mask 加一个stop action

    sample_indicator = (samples==masks.shape[1])
    samples[sample_indicator] = masks.shape[1]-1

    is_valid = torch.take_along_dim(masks, samples, dim=1)  # 判断将要删除的边是否存在，如果不存在则是invalid action。
    is_valid = is_valid.clone().bool()
    stop_action = masks.shape[1]
    samples = torch.where(is_valid, samples, stop_action)  # 如果action中都无效（=0），则最后去stop_action

    return torch.squeeze(samples, dim=1)  # squeeze 把列变成行
