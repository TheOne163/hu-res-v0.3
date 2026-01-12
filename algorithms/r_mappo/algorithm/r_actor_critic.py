import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

# MAPPO使用中心化训练（Critic使用全局状态），去中心化执行（Actor只使用局部观测）
class R_Actor(nn.Module):
    """
    用于 MAPPO 的 Actor 网络类。根据观测数据输出动作。
    :param args: (argparse.Namespace) 包含相关模型信息的参数。
    :param obs_space: (gym.Space) 观测空间。
    :param action_space: (gym.Space) 动作空间。
    :param device: (torch.device) 指定运行设备（CPU/GPU）。
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        # ==================== HU-Res-MAPPO 新增 ====================
        # 读取配置参数
        self.enable_res_hu = getattr(args, 'enable_res_hu', False)
        self.res_bias_scale = getattr(args, 'res_bias_scale', 0.1)

        if self.enable_res_hu:
            # 获取动作空间维度
            if action_space.__class__.__name__ == 'Discrete':
                self.action_dim = action_space.n
            elif action_space.__class__.__name__ == 'MultiDiscrete':
                # 对于MultiDiscrete，取所有子空间维度之和
                self.action_dim = sum(action_space.nvec)
            elif action_space.__class__.__name__ == 'Box':
                self.action_dim = action_space.shape[0]
            else:
                self.action_dim = action_space.n  # 默认fallback

            # 定义可学习的动作偏置向量 - Zero Init
            # 形状为 (1, action_dim) 以便广播到 batch 维度
            self.action_bias = nn.Parameter(torch.zeros(1, self.action_dim))

            print(f"[HU-Res-MAPPO] Action bias enabled:")
            print(f"  - Action dimension: {self.action_dim}")
            print(f"  - Bias scale (λ): {self.res_bias_scale}")
            print(f"  - Initial bias: zeros")
        # ==================== HU-Res-MAPPO 新增结束 ====================

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        根据给定的输入计算动作。
        :param obs: (np.ndarray / torch.Tensor) 网络的观测输入。
        :param rnn_states: (np.ndarray / torch.Tensor) 如果是 RNN 网络，则为 RNN 的隐藏状态。
        :param masks: (np.ndarray / torch.Tensor) 掩码张量，表示是否应将隐藏状态重新初始化为零。
        :param available_actions: (np.ndarray / torch.Tensor) 表示智能体可用的动作.（如果为 None，则所有动作都可用）
        :param deterministic: (bool) 是否从动作分布中采样或返回众数。
        :return actions: (torch.Tensor) 要执行的动作。
        :return action_log_probs: (torch.Tensor) 已执行动作的对数概率。
        :return rnn_states: (torch.Tensor) 已更新的 rnn_states。 RNN 隐藏状态。
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # ==================== HU-Res-MAPPO 修改 ====================
        if self.enable_res_hu:
            # 获取带偏置的动作（传递bias给ACTLayer）
            actions, action_log_probs = self.act(
                actor_features,
                available_actions,
                deterministic,
                action_bias=self.action_bias,  # 新增参数
                bias_scale=self.res_bias_scale  # 新增参数
            )
        else:
            # 原始逻辑
            actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        # ==================== HU-Res-MAPPO 修改结束 ====================

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        计算给定动作的对数概率和熵。
        :param obs: (torch.Tensor) 网络的观测输入。
        :param action: (torch.Tensor) 要评估熵和对数概率的动作。
        :param rnn_states: (torch.Tensor) 如果是 RNN 网络，则为 RNN 的隐藏状态。
        :param masks: (torch.Tensor) 掩码张量，表示是否应将隐藏状态重新初始化为零。
        :param available_actions: (torch.Tensor) 表示智能体可用的动作。（如果为 None，则所有动作都可用）
        :param active_masks: (torch.Tensor) 表示智能体是处于活动状态还是死亡状态。
        :return action_log_probs: (torch.Tensor) 输入动作的对数概率。
        :return dist_entropy: (torch.Tensor) 给定输入的动作分布熵。
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            # ==================== HU-Res-MAPPO 修改 ====================
            if self.enable_res_hu:
                action_log_probs, dist_entropy = self.act.evaluate_actions(
                    actor_features,
                    action,
                    available_actions,
                    active_masks=active_masks if self._use_policy_active_masks else None,
                    action_bias=self.action_bias,  # 新增参数
                    bias_scale=self.res_bias_scale  # 新增参数
                )
            else:
                action_log_probs, dist_entropy = self.act.evaluate_actions(
                    actor_features,
                    action,
                    available_actions,
                    active_masks=active_masks if self._use_policy_active_masks else None
                )
            # ==================== HU-Res-MAPPO 修改结束 ====================

        return action_log_probs, dist_entropy

    # ==================== HU-Res-MAPPO 新增方法 ====================
    def reset_action_bias(self):
        """重置动作偏置为零（全局更新后调用）"""
        if self.enable_res_hu and hasattr(self, 'action_bias'):
            self.action_bias.data.fill_(0.0)

    def get_bias_info(self):
        """获取当前偏置的统计信息（用于调试和日志）"""
        if self.enable_res_hu and hasattr(self, 'action_bias'):
            bias_data = self.action_bias.data
            return {
                'bias_mean': bias_data.mean().item(),
                'bias_std': bias_data.std().item(),
                'bias_max': bias_data.max().item(),
                'bias_min': bias_data.min().item(),
                'bias_norm': bias_data.norm().item()
            }
        return {}
    # ==================== HU-Res-MAPPO 新增方法结束 ====================


class R_Critic(nn.Module):
    """
    MAPPO 的 Critic 网络类。根据集中式输入 (MAPPO) 或本地观测 (IPPO) 输出值函数预测。
    :param args: (argparse.Namespace) 包含相关模型信息的参数。
    :param cent_obs_space: (gym.Space) （集中式）观测空间。
    :param device: (torch.device) 指定运行设备（CPU/GPU）。
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        根据给定的输入计算动作。
        :param cent_obs: (np.ndarray / torch.Tensor) 网络的观测输入。
        :param rnn_states: (np.ndarray / torch.Tensor) 如果是 RNN 网络，则为 RNN 的隐藏状态。
        :param masks: (np.ndarray / torch.Tensor) 掩码张量，表示是否应将 RNN 状态重新初始化为零。
        :return values: (torch.Tensor) 值函数预测。
        :return rnn_states: (torch.Tensor) 更新后的 RNN 隐藏状态。
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
