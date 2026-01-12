from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn
from .mlp import MLPBase # 确保引入 MLP 模块，或者直接用 nn.Sequential

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.mujoco_box = False
        self.action_type = action_space.__class__.__name__

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            self.mujoco_box = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                inputs_dim, discrete_dim, use_orthogonal, gain)])
        
        # ==================== [HU-Res-MAPPO V2 动态网络] ====================
        self.enable_res_hu = getattr(args, 'enable_res_hu', False)
        # 默认 scale，如果 forward 没传就用这个
        self.default_bias_scale = getattr(args, 'res_bias_scale', 0.1)

        if self.enable_res_hu:
            # 确定 BiasNet 的输出维度
            if self.action_type == "Discrete":
                out_dim = action_space.n
            elif self.action_type == "Box":
                out_dim = action_space.shape[0]
            else:
                out_dim = 0 # 暂不支持复杂动作空间
                print("[Warning] HU-Res Dynamic Bias currently supports Discrete/Box actions.")

            if out_dim > 0:
                # 定义动态 BiasNet (3层 MLP)
                # self.bias_net = nn.Sequential(
                #     nn.Linear(inputs_dim, 64),
                #     nn.ReLU(),
                #     nn.Linear(64, 64),
                #     nn.ReLU(),
                #     nn.Linear(64, out_dim)
                # )
                # 修改为：单层线性网络 (Linear Probe)
                # 类似于 ResNet 的最后一层，直接将特征映射为动作偏置
                self.bias_net = nn.Sequential(nn.Linear(inputs_dim,out_dim))
                
                # 初始化最后一层为 0，确保初始状态不干扰
                nn.init.constant_(self.bias_net[-1].weight, 0)
                nn.init.constant_(self.bias_net[-1].bias, 0)
                print(f"[HU-Res-MAPPO] Dynamic BiasNet initialized. In: {inputs_dim}, Out: {out_dim}")
        # ==================== [修改结束] ====================
    
    def forward(self, x, available_actions=None, deterministic=False,action_bias=None, bias_scale=None):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :param action_bias: (torch.Tensor) HU-Res-MAPPO的动作偏置 [可选]
        :param bias_scale: (float) 偏置缩放因子 λ
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.mixed_action :
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        
        elif self.mujoco_box:
            action_logits = self.action_out(x)
            
            # [HU-Res-MAPPO] 连续动作修正
            if self.enable_res_hu and hasattr(self, 'bias_net'):
                # 优先使用动态网络生成的 Bias
                # 意义：BiasNet 只看特征，但不通过梯度去修改特征。
                # 这强迫 BiasNet 适应 MainNet，而不是让 MainNet 配合 BiasNet 过拟合。
                dynamic_bias = self.bias_net(x.detach())
                # 使用传入的 scale 或默认 scale
                current_scale = bias_scale if bias_scale is not None else self.default_bias_scale
                
                # 修正均值
                new_mean = action_logits.mean + dynamic_bias * current_scale
                action_logits = type(action_logits)(new_mean, action_logits.stddev)

            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        else:
            # 【重点回答】：这里不需要传 action_bias 给 self.action_out
            # 因为 action_out 是原始的 Categorical 层，它不知道什么是 Bias
            # 我们是在它输出结果之后，手动把 Bias 加上去的
            action_logits = self.action_out(x, available_actions)
            
            # ==================== [HU-Res-MAPPO 修改] ====================
            if self.enable_res_hu and hasattr(self, 'bias_net'):
                # 1. 计算动态偏置
                dynamic_bias = self.bias_net(x)
                
                # 2. 确定缩放系数
                current_scale = bias_scale if bias_scale is not None else self.default_bias_scale
                
                # 3. 叠加 Logits
                # action_logits.logits 是主网络的原始输出
                new_logits = action_logits.logits + dynamic_bias * current_scale
                
                # 4. 重新封装分布对象
                action_logits = type(action_logits)(logits=new_logits)
            # ==================== [修改结束] ====================
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None,action_bias=None, bias_scale=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param action_bias: (torch.Tensor) HU-Res-MAPPO的动作偏置 [可选]
        :param bias_scale: (float) 偏置缩放因子 λ

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            # [修改处 4]：传递参数给 action_out
            action_logits = self.action_out(x, available_actions, 
                                            action_bias=action_bias, 
                                            bias_scale=bias_scale)
            # [HU-Res-MAPPO] 修正概率分布
            if self.enable_res_hu and hasattr(self, 'bias_net'):
                bias = self.bias_net(x)
                new_logits = action_logits.logits + bias * self.res_bias_scale
                action_logits = type(action_logits)(logits=new_logits)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, action_bias=None, bias_scale=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
                
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98 #! dosen't make sense

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
            dist_entropy = sum(dist_entropy)/len(dist_entropy)
        
        elif self.mujoco_box:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        else:
            # action_logits = self.action_out(x, available_actions)
            # ==================== HU-Res-MAPPO 修改 ====================
            action_logits = self.action_out(x, available_actions,
                                            action_bias=action_bias,
                                            bias_scale=bias_scale)

            # ==================== [HU-Res-MAPPO 修改] ====================
            if self.enable_res_hu and hasattr(self, 'bias_net'):
                dynamic_bias = self.bias_net(x)
                current_scale = bias_scale if bias_scale is not None else self.default_bias_scale
                
                new_logits = action_logits.logits + dynamic_bias * current_scale
                action_logits = type(action_logits)(logits=new_logits)
            # ==================== [修改结束] ====================

            # ==================== HU-Res-MAPPO 修改结束 ====================
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            mu_collector = []
            std_collector = []
            probs_collector = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                mu = action_logit.mean
                std = action_logit.stddev
                action_log_probs.append(action_logit.log_probs(act))
                mu_collector.append(mu)
                std_collector.append(std)
                probs_collector.append(action_logit.logits)
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_mu = torch.cat(mu_collector,-1)
            action_std = torch.cat(std_collector,-1)
            all_probs = torch.cat(probs_collector,-1)
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()
        
        else:
            action_logits = self.action_out(x, available_actions)
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            action_log_probs = action_logits.log_probs(action)
            if self.action_type=="Discrete":
                all_probs = action_logits.logits
            else:
                all_probs = None
            if active_masks is not None:
                if self.action_type=="Discrete":
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy, action_mu, action_std, all_probs