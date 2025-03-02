import torch as T
from networks_lstm import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir,
                 alpha=0.0001, beta=0.0002, fc_hidden_size=128, lstm_hidden_size=64,
                 gamma=0.99, tau=0.01, action_lb=None, action_ub=None):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_idx}'

        # 初始化策略网络和目标网络
        self.actor = ActorNetwork(alpha, actor_dims, lstm_hidden_size, fc_hidden_size, n_actions,
                                  action_lb=action_lb, action_ub=action_ub,
                                  chkpt_dir=chkpt_dir, name=self.agent_name + '_actor')
        self.target_actor = ActorNetwork(alpha, actor_dims, lstm_hidden_size, fc_hidden_size, n_actions,
                                         action_lb=action_lb, action_ub=action_ub,
                                         chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor')

        # 初始化价值网络和目标价值网络
        self.critic = CriticNetwork(beta, critic_dims, lstm_hidden_size, fc_hidden_size,
                                    n_agents=1, n_actions=n_actions,
                                    chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.target_critic = CriticNetwork(beta, critic_dims, lstm_hidden_size, fc_hidden_size,
                                           n_agents=1, n_actions=n_actions,
                                           chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic')

        # 初始化目标网络参数为主网络参数
        self.update_network_parameters(tau=1)

    def choose_action(self, obs_history, time_step, evaluate=False):
        """
        根据历史观测值，结合策略网络输出一个动作，并添加噪声用于探索。
        """
        obs_history = T.tensor([obs_history], dtype=T.float).to(self.actor.device)

        # 使用 Actor 网络预测动作
        actions = self.actor.forward(obs_history)

        # 噪声参数
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        # 计算噪声尺度
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1  # [-1, 1) 区间的随机噪声
        if not evaluate:
            noise = noise_scale * noise  # 添加噪声
        else:
            noise = 0 * noise  # 评估模式下不添加噪声

        action = actions + noise

        # 动作范围限制
        if self.actor.action_lb is not None and self.actor.action_ub is not None:
            action = T.clamp(action, self.actor.action_lb, self.actor.action_ub)

        action_np = action.detach().cpu().numpy()[0]
        return action_np

    def update_network_parameters(self, tau=None):
        """
        使用软更新方法更新目标网络的参数。
        """
        if tau is None:
            tau = self.tau

        # 更新目标策略网络参数
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 更新目标价值网络参数
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        """保存模型参数到文件。"""
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        """从文件加载模型参数。"""
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

