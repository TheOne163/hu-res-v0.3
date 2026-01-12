import torch
import numpy as np
import torch.nn.functional as F
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    用于存储训练数据的缓冲区。
    :param args: (argparse.Namespace) 包含相关模型、策略和环境信息的参数。
    :param num_agents: (int) 环境中的智能体数量。
    :param obs_space: (gym.Space) 智能体的观测空间。
    :param cent_obs_space: (gym.Space) 智能体的集中式观测空间。
    :param act_space: (gym.Space) 智能体的动作空间。
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

        # HU-MAPPO 支持
        self.enable_hu = getattr(args, 'enable_hu', False)
        self.sub_episode_length = getattr(args, 'sub_episode_length', 100)

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        将数据插入缓冲区。
        :param share_obs: (argparse.Namespace) 包含相关模型、策略和环境信息的参数。
        :param obs: (np.ndarray) 本地智能体观测值。
        :param rnn_states_actor: (np.ndarray) Actor 网络的 RNN 状态。
        :param rnn_states_critic: (np.ndarray) Critic 网络的 RNN 状态。
        :param actions: (np.ndarray) 智能体采取的动作。
        :param action_log_probs: (np.ndarray) 智能体采取动作的对数概率。
        :param value_preds: (np.ndarray) 每一步的值函数预测值。
        :param rewards: (np.ndarray) 每一步收集的奖励。
        :param mask: (np.ndarray) 表示环境是否已终止。
        :param bad_masks: (np.ndarray) 智能体的动作空间。
        :param active_masks: (np.ndarray) 表示智能体在环境中是处于活动状态还是死亡状态。
        :param available_actions: (np.ndarray) 每个智能体可用的动作。如果为 None，则所有动作都可用。
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        将数据插入缓冲区。此插入函数专用于回合制的 Hanabi 算法。
        :param share_obs: (argparse.Namespace) 包含相关模型、策略和环境信息的参数。
        :param obs: (np.ndarray) 本地智能体观测值。
        :param rnn_states_actor: (np.ndarray) Actor 网络的 RNN 状态。
        :param rnn_states_critic: (np.ndarray) Critic 网络的 RNN 状态。
        :param actions: (np.ndarray) 智能体采取的动作。
        :param action_log_probs: (np.ndarray) 智能体采取动作的对数概率。
        :param value_preds: (np.ndarray) 每一步的值函数预测值。
        :param rewards: (np.ndarray) 每一步收集的奖励。
        :param mask: (np.ndarray) 表示环境是否已终止。
        :param bad_masks: (np.ndarray) 表示环境是否真正终止，还是由于回合数限制而终止。
        :param active_masks: (np.ndarray) 表示智能体在环境中是处于活动状态还是死亡状态。
        :param available_actions: (np.ndarray) 每个智能体可用的动作。如果为 None，则所有动作都可用。
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        """将最后一个时间步的数据复制到第一个索引位置。此操作在模型更新后调用。"""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        """将最后一个时间步的数据复制到第一个索引处。此方法用于 Hanabi 数据库。"""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        计算回报的方式有两种：一种是奖励的折扣总和，另一种是使用 GAE（广义平均方程）。
        :param next_value: (np.ndarray) 上一回合之后下一回合的预测值。
        :param value_normalizer: (PopArt) 如果不为 None，则为 PopArt 值归一化器实例。
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        if self.algo == "mat" or self.algo == "mat_dec":
                            value_t = value_normalizer.denormalize(self.value_preds[step])
                            value_t_next = value_normalizer.denormalize(self.value_preds[step + 1])
                            rewards_t = self.rewards[step]

                            # mean_v_t = np.mean(value_t, axis=-2, keepdims=True)
                            # mean_v_t_next = np.mean(value_t_next, axis=-2, keepdims=True)
                            # delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                            delta = rewards_t + self.gamma * self.masks[step + 1] * value_t_next - value_t
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.advantages[step] = gae
                            self.returns[step] = gae + value_t
                        else:
                            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                                self.value_preds[step + 1]) * self.masks[step + 1] \
                                    - value_normalizer.denormalize(self.value_preds[step])
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        if self.algo == "mat" or self.algo == "mat_dec":
                            rewards_t = self.rewards[step]
                            mean_v_t = np.mean(self.value_preds[step], axis=-2, keepdims=True)
                            mean_v_t_next = np.mean(self.value_preds[step + 1], axis=-2, keepdims=True)
                            delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                            # delta = rewards_t + self.gamma * self.value_preds[step + 1] * \
                            #         self.masks[step + 1] - self.value_preds[step]
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.advantages[step] = gae
                            self.returns[step] = gae + self.value_preds[step]

                        else:
                            delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                                    self.masks[step + 1] - self.value_preds[step]
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        生成用于 MLP 策略的训练数据。
        :param benefits: (np.ndarray) 优势估计值
        :param num_mini_batch: (int) 将批次分割成的 minibatch 数量。
        :param mini_batch_size: (int) 每个 minibatch 中的样本数量。
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Yield training data for non-chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    # ==================== 添加以下方法 ====================
    def compute_sub_returns(self, sub_start, sub_end, next_value, gamma, gae_lambda, value_normalizer=None):
        """
        计算子episode的returns和advantages (修复版)
        """
        sub_length = sub_end - sub_start

        # 1. 准备 Value Predictions
        # 获取当前段的 value_preds，如果是归一化的，统一先反归一化为真实值
        if self._use_popart or self._use_valuenorm:
            # value_preds 切片包含 [sub_start, sub_end]，长度为 sub_length + 1
            value_preds = value_normalizer.denormalize(self.value_preds[sub_start:sub_end + 1])
            # bootstrap value 也是归一化的，需要反归一化
            next_value = value_normalizer.denormalize(next_value)
        else:
            value_preds = self.value_preds[sub_start:sub_end + 1].copy()

        # 2. 设置 Bootstrap (self.returns 存储真实值)
        self.returns[sub_end] = next_value

        # 3. 反向计算 GAE
        gae = 0
        for step in reversed(range(sub_length)):
            actual_step = sub_start + step

            # GAE Delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            # 我们使用 value_preds 作为 V 的估计，而不是使用 self.returns

            # 获取 V(s_{t+1})
            if step == sub_length - 1:
                v_next = next_value
            else:
                v_next = value_preds[step + 1]  # 这里的 step+1 对应切片中的索引

            # 计算 Delta (使用真实值计算)
            delta = (self.rewards[actual_step]
                     + gamma * v_next * self.masks[actual_step + 1]
                     - value_preds[step])

            # 计算 GAE
            gae = delta + gamma * gae_lambda * self.masks[actual_step + 1] * gae

            # 存储 Return = GAE + V(s_t)
            # 这里的 Return 是真实值 (Denormalized)
            self.returns[actual_step] = gae + value_preds[step]
            self.advantages[actual_step] = gae

    def sub_feed_forward_generator(self, sub_start, sub_end, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        为子episode生成feed-forward训练数据

        Args:
            sub_start: 子episode起始step
            sub_end: 子episode结束step（不含）
            advantages: 归一化后的advantages
            num_mini_batch: mini batch数量
            mini_batch_size: 每个mini batch大小

        Yields:
            训练数据batch的tuple
        """
        sub_length = sub_end - sub_start
        batch_size = self.n_rollout_threads * sub_length * self.num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"batch_size ({batch_size}) < num_mini_batch ({num_mini_batch})"
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size]
                   for i in range(num_mini_batch)]

        # 提取子episode数据并reshape
        share_obs = self.share_obs[sub_start:sub_end].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[sub_start:sub_end].reshape(-1, *self.obs.shape[3:])

        rnn_states = self.rnn_states[sub_start:sub_end].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[sub_start:sub_end].reshape(-1, *self.rnn_states_critic.shape[3:])

        actions = self.actions[sub_start:sub_end].reshape(-1, self.actions.shape[-1])

        if self.available_actions is not None:
            available_actions = self.available_actions[sub_start:sub_end].reshape(-1, self.available_actions.shape[-1])
        else:
            available_actions = None

        value_preds = self.value_preds[sub_start:sub_end].reshape(-1, 1)
        returns = self.returns[sub_start:sub_end].reshape(-1, 1)
        masks = self.masks[sub_start:sub_end].reshape(-1, 1)
        active_masks = self.active_masks[sub_start:sub_end].reshape(-1, 1)
        action_log_probs = self.action_log_probs[sub_start:sub_end].reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, \
                actions_batch, value_preds_batch, return_batch, masks_batch, \
                active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def sub_recurrent_generator(self, sub_start, sub_end, advantages, num_mini_batch, data_chunk_length):
        """
        为子episode生成RNN训练数据

        Args:
            sub_start: 子episode起始step
            sub_end: 子episode结束step（不含）
            advantages: 归一化后的advantages
            num_mini_batch: mini batch数量
            data_chunk_length: RNN序列长度

        Yields:
            训练数据batch的tuple
        """
        sub_length = sub_end - sub_start

        # 调整chunk_length以适应子episode长度
        episode_length = sub_length
        effective_chunk_length = min(data_chunk_length, episode_length)

        data_chunks = episode_length // effective_chunk_length
        batch_size = self.n_rollout_threads * self.num_agents * data_chunks

        if batch_size < num_mini_batch:
            mini_batch_size = 1
            num_mini_batch = batch_size
        else:
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size]
                   for i in range(num_mini_batch)]

        # 提取子episode数据
        share_obs = self.share_obs[sub_start:sub_end]
        obs = self.obs[sub_start:sub_end]
        actions = self.actions[sub_start:sub_end]
        value_preds = self.value_preds[sub_start:sub_end]
        returns = self.returns[sub_start:sub_end]
        masks = self.masks[sub_start:sub_end]
        active_masks = self.active_masks[sub_start:sub_end]
        action_log_probs = self.action_log_probs[sub_start:sub_end]
        rnn_states = self.rnn_states[sub_start:sub_end]
        rnn_states_critic = self.rnn_states_critic[sub_start:sub_end]

        if self.available_actions is not None:
            available_actions = self.available_actions[sub_start:sub_end]
        else:
            available_actions = None

        # 分chunk处理
        share_obs_batch = []
        obs_batch = []
        rnn_states_batch = []
        rnn_states_critic_batch = []
        actions_batch = []
        available_actions_batch = []
        value_preds_batch = []
        return_batch = []
        masks_batch = []
        active_masks_batch = []
        old_action_log_probs_batch = []
        adv_targ_batch = []

        for thread_id in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                for chunk_idx in range(data_chunks):
                    start_idx = chunk_idx * effective_chunk_length
                    end_idx = start_idx + effective_chunk_length

                    share_obs_batch.append(share_obs[start_idx:end_idx, thread_id, agent_id])
                    obs_batch.append(obs[start_idx:end_idx, thread_id, agent_id])
                    actions_batch.append(actions[start_idx:end_idx, thread_id, agent_id])
                    if available_actions is not None:
                        available_actions_batch.append(available_actions[start_idx:end_idx, thread_id, agent_id])
                    value_preds_batch.append(value_preds[start_idx:end_idx, thread_id, agent_id])
                    return_batch.append(returns[start_idx:end_idx, thread_id, agent_id])
                    masks_batch.append(masks[start_idx:end_idx, thread_id, agent_id])
                    active_masks_batch.append(active_masks[start_idx:end_idx, thread_id, agent_id])
                    old_action_log_probs_batch.append(action_log_probs[start_idx:end_idx, thread_id, agent_id])
                    adv_targ_batch.append(advantages[start_idx:end_idx, thread_id, agent_id])
                    rnn_states_batch.append(rnn_states[start_idx, thread_id, agent_id])
                    rnn_states_critic_batch.append(rnn_states_critic[start_idx, thread_id, agent_id])

        # Stack所有chunks
        T, N = effective_chunk_length, self.n_rollout_threads * self.num_agents * data_chunks

        share_obs_batch = np.stack(share_obs_batch)
        obs_batch = np.stack(obs_batch)
        actions_batch = np.stack(actions_batch)
        if len(available_actions_batch) > 0:
            available_actions_batch = np.stack(available_actions_batch)
        else:
            available_actions_batch = None
        value_preds_batch = np.stack(value_preds_batch)
        return_batch = np.stack(return_batch)
        masks_batch = np.stack(masks_batch)
        active_masks_batch = np.stack(active_masks_batch)
        old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
        adv_targ_batch = np.stack(adv_targ_batch)
        rnn_states_batch = np.stack(rnn_states_batch)
        rnn_states_critic_batch = np.stack(rnn_states_critic_batch)

        for indices in sampler:
            L = len(indices)

            share_obs_b = share_obs_batch[indices].reshape(L * T, *share_obs_batch.shape[2:])
            obs_b = obs_batch[indices].reshape(L * T, *obs_batch.shape[2:])
            actions_b = actions_batch[indices].reshape(L * T, -1)
            if available_actions_batch is not None:
                available_actions_b = available_actions_batch[indices].reshape(L * T, -1)
            else:
                available_actions_b = None
            value_preds_b = value_preds_batch[indices].reshape(L * T, 1)
            return_b = return_batch[indices].reshape(L * T, 1)
            masks_b = masks_batch[indices].reshape(L * T, 1)
            active_masks_b = active_masks_batch[indices].reshape(L * T, 1)
            old_action_log_probs_b = old_action_log_probs_batch[indices].reshape(L * T, -1)
            adv_targ_b = adv_targ_batch[indices].reshape(L * T, 1)
            rnn_states_b = rnn_states_batch[indices]
            rnn_states_critic_b = rnn_states_critic_batch[indices]

            yield share_obs_b, obs_b, rnn_states_b, rnn_states_critic_b, actions_b, \
                value_preds_b, return_b, masks_b, active_masks_b, old_action_log_probs_b, \
                adv_targ_b, available_actions_b
    # ==================== 方法添加结束 ====================