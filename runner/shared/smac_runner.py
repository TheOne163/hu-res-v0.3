import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    """训练主循环，协助环境交互和策略更新"""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        # ==================== 在 __init__ 中添加 ====================
        # HU-MAPPO 配置
        self.enable_hu = self.all_args.enable_hu
        if self.enable_hu:
            self.sub_episode_length = self.all_args.sub_episode_length
            self.num_sub_episodes = self.episode_length // self.sub_episode_length

            # 验证配置
            assert self.episode_length % self.sub_episode_length == 0, \
                f"episode_length ({self.episode_length}) must be divisible by " \
                f"sub_episode_length ({self.sub_episode_length})"

            print(f"[HU-MAPPO] Enabled: {self.num_sub_episodes} sub-episodes × "
                  f"{self.sub_episode_length} steps = {self.episode_length} steps/episode")
            print(f"[HU-MAPPO] Sub-update: {self.all_args.sub_ppo_epoch} epochs, "
                  f"lr_scale={self.all_args.sub_lr_scale}")
        # ==================== 添加结束 ====================

    """
    run()
      → warmup()              # 预热，获取初始观测
      → collect()             # 收集经验
      → compute()             # 计算优势和回报
      → train()               # 更新策略
      → save()                # 保存模型
      → log()                 # 记录日志
    """
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # for step in range(self.episode_length):
            #     # Sample actions
            #     values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
            #
            #     # Obser reward and next obs
            #     obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
            #
            #     data = obs, share_obs, rewards, dones, infos, available_actions, \
            #            values, actions, action_log_probs, \
            #            rnn_states, rnn_states_critic
            #
            #     # insert data into buffer
            #     self.insert(data)
            # ========== HU-MAPPO 分支 ==========
            if self.enable_hu:
                # 使用分层更新收集数据
                sub_train_infos,infos = self.collect_with_hu()
            else:
                # 原始数据收集
                for step in range(self.episode_length):
                    # 收集数据
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = \
                        self.collect(step)
                    actions_env = actions  # 在SMAC中，actions可以直接作为环境输入

                    # 环境step
                    # 2. 环境执行 (修正：接收 6 个返回值)
                    obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                    # 3. 打包数据 (修正：包含 share_obs 和 available_actions)
                    # 注意：这里的数据顺序必须与 insert 方法中的解包顺序完全一致
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                        values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic

                    # 插入buffer
                    self.insert(data)
            # ====================================

            # compute return and update network
            self.compute()      #计算优势函数
            train_infos = self.train()      #策略更新
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):      #保存模型
                self.save()

            # log information
            if episode % self.log_interval == 0:        #保存日志文件

                # 添加HU相关日志
                if self.enable_hu and hasattr(self, 'last_sub_train_infos'):
                    train_infos.update(self.last_sub_train_infos)

                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2" or self.env_name == "SMACv2" or self.env_name == "SMAC" or self.env_name == "StarCraft2v2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):       #重置环境，获取初始观测、状态、可用动作
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()


    """调用策略获取动作、执行动作获取下一状态和奖励、存储经验到缓冲区
    collect(step)
      → policy.get_actions()     # 策略采样动作
      → envs.step()              # 环境执行动作
      → buffer.insert()          # 存储经验
    """
    @torch.no_grad()
    def collect(self, step):        #在当前步收集经验数据
        self.trainer.prep_rollout()
        # 1. 获取动作和价值
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        # 3. 存储到缓冲区
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            else:
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break

    def collect_with_hu(self):
        """
        【HU-MAPPO核心】带分层更新的数据收集

        Returns:
            sub_train_infos: 子更新训练统计列表
        """
        sub_train_infos = []

        for sub_ep in range(self.num_sub_episodes):
            sub_start = sub_ep * self.sub_episode_length
            sub_end = (sub_ep + 1) * self.sub_episode_length

            # 收集子episode数据
            for step in range(sub_start, sub_end):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, = \
                    self.collect(step)
                actions_env = actions# 在SMAC中，actions可以直接作为环境输入

                # 2. 环境执行 (修正：接收 6 个返回值)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # 3. 打包数据 (修正：包含 share_obs 和 available_actions)
                # 注意：这里的数据顺序必须与 insert 方法中的解包顺序完全一致
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic

                self.insert(data)

            # 子更新（最后一个子episode留给完整更新）
            if sub_ep < self.num_sub_episodes - 1:
                sub_info = self.sub_update(sub_start, sub_end)
                sub_train_infos.append(sub_info)

        # 保存子更新统计（供日志使用）
        if sub_train_infos:
            self.last_sub_train_infos = self._aggregate_sub_train_infos(sub_train_infos)

        return sub_train_infos, infos

    def sub_update(self, sub_start, sub_end):
        """
        【HU-MAPPO核心】执行子episode更新

        Args:
            sub_start: 子episode起始step
            sub_end: 子episode结束step（不含）

        Returns:
            sub_train_info: 子更新训练统计
        """
        # 获取bootstrap value
        self.trainer.prep_rollout()

        # 获取下一时刻的value估计
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[sub_end]),
            np.concatenate(self.buffer.rnn_states_critic[sub_end]),
            np.concatenate(self.buffer.masks[sub_end])
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        # 计算子episode的returns
        self.buffer.compute_sub_returns(
            sub_start,
            sub_end,
            next_values,
            self.all_args.gamma,
            self.all_args.gae_lambda,
            self.trainer.value_normalizer if self.all_args.use_popart or self.all_args.use_valuenorm else None
        )

        # 执行子训练
        self.trainer.prep_training()
        sub_train_info = self.trainer.sub_train(self.buffer, sub_start, sub_end)

        return sub_train_info

    def _aggregate_sub_train_infos(self, sub_train_infos):
        """聚合多个子更新的训练统计"""
        aggregated = {}
        for key in sub_train_infos[0].keys():
            aggregated[key] = np.mean([info[key] for info in sub_train_infos])
        return aggregated