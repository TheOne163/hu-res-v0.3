import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO策略类。封装了actor网络和critic网络，用于计算动作和价值函数预测。
    :param args: (argparse.Namespace) 参数，包含相关的模型和策略信息。
    :param obs_space: (gym.Space) 观测空间。
    :param cent_obs_space: (gym.Space) 价值函数输入空间（MAPPO使用集中式输入，IPPO使用分散式输入）。
    :param action_space: (gym.Space) 动作空间。
    :param device: (torch.device) 指定运行设备（CPU/GPU）。
        __init__()
      → R_Actor()   # 创建策略网络
      → R_Critic()  # 创建价值网络
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        #分别是带 RNN 的 actor 和 critic 类（“R_” 表示 recurrent）
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # 分别为 actor 和 critic 创建 Adam 优化器，分别独立更新（方便设置不同 lr）。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.降低 Actor 模型和 Critic 模型的学习率。
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        按线性方式把 learning rate 从初始 lr 逐步衰减到 0（或某个下限），常用于稳定训练。
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        计算给定输入对应的动作和值函数预测。
        :param cent_obs (np.ndarray): 集中式输入到评论家。
        :param obs (np.ndarray): 局部智能体输入到执行器。
        :param rnn_states_actor: (np.ndarray) 如果 actor 是 RNN，则为 actor 的 RNN 状态。
        :param rnn_states_critic: (np.ndarray) 如果 critic 是 RNN，则为 critic 的 RNN 状态。
        :param mask: (np.ndarray) 表示 RNN 状态应该重置的点。
        :param available_actions: (np.ndarray) 表示智能体可用的动作。（如果为 None，则表示所有动作都可用）
        :param deterministic: (bool) 动作应该是按分布模式还是按采样模式。
        :return values: (torch.Tensor) 值函数预测。
        返回 actions：(torch.Tensor) 要执行的动作。
        返回 action_log_probs：(torch.Tensor) 已选择动作的概率对数。
        返回 rnn_states_actor：(torch.Tensor) 更新后的 Actor 网络 RNN 状态。
        返回 rnn_states_critic：(torch.Tensor) 更新后的 Critic 网络 RNN 状态。
         get_actions()
             → actor.forward()     # 获取动作分布
             → critic.forward()    # 获取价值估计
             → action_dist.sample()  # 采样动作
             → action_dist.log_probs()  # 计算log概率
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        获取值函数预测结果。
        :param cent_obs (np.ndarray): 集中式输入到评论器。
        :param rnn_states_critic: (np.ndarray) 如果评论器是 RNN，则为评论器的 RNN 状态。
        :param mask: (np.ndarray) 表示 RNN 状态应重置的点。
        :return values: (torch.Tensor) 值函数预测结果。
          get_values()
            → critic.forward()  # 价值估计
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        获取用于 Actor 更新的动作对数概率/熵和价值函数预测。
        :param cent_obs (np.ndarray): 集中式输入到评论家。
        :param obs (np.ndarray): 局部智能体输入到 Actor。
        :param rnn_states_actor: (np.ndarray) 如果 Actor 是 RNN，则为 Actor 的 RNN 状态。
        :param rnn_states_critic: (np.ndarray) 如果评论家是 RNN，则为评论家的 RNN 状态。
        :param action: (np.ndarray) 要计算其对数概率和熵的动作。
        :param mask: (np.ndarray) 表示 RNN 状态应重置的点。
        :param available_actions: (np.ndarray) 表示智能体可用的动作。（如果为 None，则表示所有动作都可用）
        :param active_masks: (torch.Tensor) 表示智能体处于激活状态还是死亡状态。
        :return values: (torch.Tensor) 值函数预测结果。
        :return action_log_probs: (torch.Tensor) 输入动作的对数概率。
        :return dist_entropy: (torch.Tensor) 给定输入的动作分布熵。
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        根据给定的输入计算动作。
        :param obs (np.ndarray): 本地智能体输入到 Actor 中。
        :param rnn_states_actor: (np.ndarray) 如果 Actor 是 RNN，则为 Actor 的 RNN 状态。
        :param mask: (np.ndarray) 表示 RNN 状态应该重置的点。
        :param available_actions: (np.ndarray) 表示智能体可用的动作。（如果为 None，则所有动作都可用）
        :param deterministic: (bool) 表示该动作是应该使用分布模式还是应该进行采样。
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
