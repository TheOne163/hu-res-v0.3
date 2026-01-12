import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None, action_bias=None, bias_scale=0.1):
        """
        前向传播计算动作分布

        :param x: (torch.Tensor) 输入特征 [batch_size, hidden_size]
        :param available_actions: (torch.Tensor) 可用动作掩码 [batch_size, num_actions]
        :param action_bias: (torch.Tensor) HU-Res-MAPPO的动作偏置 [1, num_actions] 或 None
        :param bias_scale: (float) 偏置缩放因子 λ，默认 0.1
        :return: FixedCategorical 分布对象
        """
        # Step 1: 计算基础logits
        x = self.linear(x)  # [batch_size, num_actions]

        # ==================== HU-Res-MAPPO 核心修改 ====================
        # Step 2: 添加动作偏置（在mask之前）
        if action_bias is not None:
            # action_bias 形状: [1, num_actions]
            # x 形状: [batch_size, num_actions]
            # 广播机制自动处理batch维度
            x = x + bias_scale * action_bias
        # ==================== HU-Res-MAPPO 核心修改结束 ====================

        # Step 3: 应用动作掩码
        if available_actions is not None:
            # 将不可用动作的logits设为极小值
            x[available_actions == 0] = -1e10

        # Step 4: 创建并返回分布
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    # def forward(self, x):
    #     action_mean = self.fc_mean(x)
    #
    #     #  An ugly hack for my KFAC implementation.
    #     zeros = torch.zeros(action_mean.size())
    #     if x.is_cuda:
    #         zeros = zeros.cuda()
    #
    #     action_logstd = self.logstd(zeros)
    #     return FixedNormal(action_mean, action_logstd.exp())
    def forward(self, x, available_actions=None, action_bias=None, bias_scale=0.1):
        """
        前向传播计算动作分布

        注意：对于连续动作空间，action_bias会加到均值上
        """
        action_mean = self.fc_mean(x)

        # ==================== HU-Res-MAPPO 修改（连续动作版本）====================
        if action_bias is not None:
            action_mean = action_mean + bias_scale * action_bias
        # ==================== HU-Res-MAPPO 修改结束 ====================

        action_std = self.logstd(torch.zeros_like(action_mean))
        return FixedNormal(action_mean, action_std.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    # def forward(self, x):
    #     x = self.linear(x)
    #     return FixedBernoulli(logits=x)
    def forward(self, x, available_actions=None, action_bias=None, bias_scale=0.1):
        """
        前向传播计算动作分布
        """
        x = self.linear(x)

        # ==================== HU-Res-MAPPO 修改 ====================
        if action_bias is not None:
            x = x + bias_scale * action_bias
        # ==================== HU-Res-MAPPO 修改结束 ====================

        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
