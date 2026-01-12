import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    #实现MAPPO的策略网络和更新逻辑
    """
    用于 MAPPO 更新策略的训练器类。
    :param args: (argparse.Namespace) 参数，包含相关的模型、策略和环境信息。
    :param policy: (R_MAPPO_Policy) 要更新的策略。
    :param device: (torch.device) 指定运行设备（CPU/GPU）。
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # ==================== HU-Res-MAPPO 配置 ====================
        self.enable_res_hu = getattr(args, 'enable_res_hu', False)
        # 读取子更新配置
        self.enable_hu = getattr(args, 'enable_hu', False)
        self.sub_ppo_epoch = getattr(args, 'sub_ppo_epoch', 3)
        self.sub_num_mini_batch = getattr(args, 'sub_num_mini_batch', 1)
        self.sub_lr_scale = getattr(args, 'sub_lr_scale', 1.0)
        self.sub_entropy_coef = getattr(args, 'sub_entropy_coef', 0.01)
        self.sub_value_loss_coef = getattr(args, 'sub_value_loss_coef', 0.5)
        # 动态切换Clip参数的配置
        self.sub_clip_param = getattr(args, 'sub_clip_param', 0.2)
        
        if self.enable_res_hu:
            print(f"[HU-Res-MAPPO] Dynamic BiasNet Framework initialized.")
        # ==================== 配置结束 ====================

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None


    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算值函数损失。
        :param values: (torch.Tensor) 值函数预测值。
        :param value_preds_batch: (torch.Tensor) 来自数据批次的“旧”值预测值（用于值裁剪损失）。
        :param return_batch: (torch.Tensor) 返回的奖励批次。
        :param active_masks_batch: (torch.Tensor) 表示智能体在给定时间戳 (timesep) 时是处于激活状态还是死亡状态。
        :return value_loss: (torch.Tensor) 值函数损失。
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True, value_loss_coef=None, entropy_coef=None):
        """
        更新 Actor 和 Critic 网络。
        :param sample: (Tuple) 包含用于更新网络的数据批次。
        :update_actor: (bool) 是否更新 Actor 网络。
        :return value_loss: (torch.Tensor) 值函数损失。
        :return critic_grad_norm: (torch.Tensor) 来自 Critic 更新的梯度归一化。
        :return policy_loss: (torch.Tensor) Actor（策略）损失值。
        :return dist_entropy: (torch.Tensor) 动作熵。
        :return actor_grad_norm: (torch.Tensor) 来自 Actor 更新的梯度归一化。
        :return imp_weights: (torch.Tensor) 重要性采样权重。
        """
        if value_loss_coef is None:
            value_loss_coef = self.value_loss_coef
        if entropy_coef is None:
            entropy_coef = self.entropy_coef
        # 1. 解包样本数据
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 调整为在一次前向传递中完成所有步骤
        # 2. 前向传播获取当前策略
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        # 4. 计算策略损失（PPO clip目标）
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),dim=-1,keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        
        self.policy.actor_optimizer.zero_grad()

        # 6. 反向传播和优化
        """
        更新 Actor 和 Critic 网络。
        :param sample: (Tuple) 包含用于更新网络的数据批次。
        :update_actor: (bool) 是否更新 Actor 网络。
        :return value_loss: (torch.Tensor) 值函数损失。
        :return critic_grad_norm: (torch.Tensor) 来自 Critic 更新的梯度归一化。
        :return policy_loss: (torch.Tensor) Actor（策略）损失值。
        :return dist_entropy: (torch.Tensor) 动作熵。
        :return actor_grad_norm: (torch.Tensor) 来自 Actor 更新的梯度归一化。
        :return imp_weights: (torch.Tensor) 重要性采样权重。
               """
        """optimizer.zero_grad();loss.backward();optimizer.step();self.policy.actor_optimizer.zero_grad(),
        梯度清零、反向传播和参数更新
        """


        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        # 3. 计算价值损失
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        # 6. 反向传播和优化
        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        train(buffer)
            → buffer.feed_forward_generator()  # 生成mini-batch
            → ppo_update(sample)                # 更新每个batch
        使用小批量梯度下降法执行训练更新。
        :param buffer: (SharedReplayBuffer) 包含训练数据的缓冲区。
        :param update_actor: (bool) 是否更新 Actor 网络。
        :return train_info: (dict) 包含有关训练更新的信息（例如损失、梯度范数等）。
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                """value_loss 是一个 tensor,value_loss.item() 将其转换为 普通 Python number（float）。
                ppo_update,每执行一次 mini-batch 更新，就把这个更新的 value loss 加到总的损失中"""
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm.item() if isinstance(critic_grad_norm, torch.Tensor) else critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        #循环结束以后再除以更新次数：就变成当前 epoch 平均每次更新的损失。
        for k in train_info.keys():
            train_info[k] /= num_updates

  # ==================== HU-Res-MAPPO 修改 ====================
        # 【重要】移除了所有的重置 (reset) 和衰减 (decay) 逻辑
        # BiasNet 现在是一个持续学习的神经网络，不需要在全局更新后重置。
        if self.enable_res_hu:
             print("BiasNet 在工作")
             pass  
        # ==================== HU-Res-MAPPO 核心修改结束 ====================
 
        return train_info

    # ==================== HU-Res-MAPPO 核心新增方法 ====================
    # ==================== HU-Res-MAPPO 核心新增方法 ====================
    def sub_train(self, buffer, sub_start, sub_end):
        """
        【HU-Res-MAPPO核心】执行子更新训练
        """
        # ==================== Step 1: 冻结主网络参数 ====================
        frozen_params_count = 0
        trainable_params_count = 0

        if self.enable_res_hu:
            # 遍历 Actor 网络参数
            for name, param in self.policy.actor.named_parameters():
                # 【修改】只训练 bias_net
                # 之前是 'action_bias' (vector)，现在是 'bias_net' (module)
                if 'bias_net' in name:
                    param.requires_grad = True
                    trainable_params_count += param.numel()
                else:
                    param.requires_grad = False
                    frozen_params_count += param.numel()

            print(f"[HU-Res Sub] Freezing MainNet. Trainable (BiasNet): {trainable_params_count}, Frozen: {frozen_params_count}")
        # ==================== Step 1 结束 ====================

        # ==================== Step 2: 计算优势函数 ====================
        if self._use_popart or self._use_valuenorm:
            sub_advantages = buffer.returns[sub_start:sub_end] - \
                             self.value_normalizer.denormalize(buffer.value_preds[sub_start:sub_end])
        else:
            sub_advantages = buffer.returns[sub_start:sub_end] - buffer.value_preds[sub_start:sub_end]

        sub_advantages_copy = sub_advantages.copy()
        sub_advantages_copy[buffer.active_masks[sub_start:sub_end] == 0.0] = np.nan
        mean_adv = np.nanmean(sub_advantages_copy)
        std_adv = np.nanstd(sub_advantages_copy)
        sub_advantages = (sub_advantages - mean_adv) / (std_adv + 1e-5)

        # ==================== Step 3: 执行子更新PPO训练 ====================
        train_info = {}
        train_info['sub_value_loss'] = 0
        train_info['sub_policy_loss'] = 0
        train_info['sub_dist_entropy'] = 0
        train_info['sub_actor_grad_norm'] = 0
        train_info['sub_ratio'] = 0
        
        # 临时切换 Clip 参数，允许 BiasNet 大幅修正
        original_clip = self.clip_param
        if hasattr(self, 'sub_clip_param'):
            self.clip_param = self.sub_clip_param

        sub_ppo_epoch = self.sub_ppo_epoch

        for _ in range(sub_ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.sub_recurrent_generator(
                    sub_start, sub_end, sub_advantages,
                    self.num_mini_batch, self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.sub_naive_recurrent_generator(
                    sub_start, sub_end, sub_advantages, self.num_mini_batch
                )
            else:
                data_generator = buffer.sub_feed_forward_generator(
                    sub_start, sub_end, sub_advantages, self.num_mini_batch
                )

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor=True)

                train_info['sub_value_loss'] += value_loss.item()
                train_info['sub_policy_loss'] += policy_loss.item()
                train_info['sub_dist_entropy'] += dist_entropy.item()
                train_info['sub_actor_grad_norm'] += actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm
                train_info['sub_ratio'] += imp_weights.mean().item()

        num_updates = sub_ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        
        # 恢复 Clip 参数
        self.clip_param = original_clip
        # ==================== Step 3 结束 ====================

        # ==================== Step 4: 解冻所有参数 ====================
        if self.enable_res_hu:
            for param in self.policy.actor.parameters():
                param.requires_grad = True

            print(f"[HU-Res Sub] Completed. MainNet Unfrozen.")
        # ==================== Step 4 结束 ====================

        return train_info
    
    # ==================== HU-Res-MAPPO 核心新增方法结束 ====================

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
