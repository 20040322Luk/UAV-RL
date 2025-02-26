import torch as T
from networks_lstm import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                 alpha=0.0001, beta=0.0002, fc1=128, fc2=128, lstm_hidden_size=64,
                 gamma=0.99, tau=0.01):
        """
        初始化智能体，包括策略网络、目标网络和相关参数。
        :param actor_dims: 策略网络输入维度
        :param critic_dims: 价值网络输入维度
        :param n_actions: 动作空间维度
        :param n_agents: 智能体数量
        :param agent_idx: 智能体索引
        :param chkpt_dir: 检查点保存路径
        :param alpha: 策略网络学习率
        :param beta: 价值网络学习率
        :param fc1: 第一层全连接层神经元数量
        :param fc2: 第二层全连接层神经元数量
        :param lstm_hidden_size: LSTM隐藏层大小（新增参数）
        :param gamma: 折扣因子
        :param tau: 软更新系数
        """
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_idx}'

        # 初始化策略网络和目标网络
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, lstm_hidden_size,
                                  chkpt_dir=chkpt_dir, name=self.agent_name + '_actor')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, lstm_hidden_size,
                                         chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor')

        # 初始化价值网络和目标价值网络
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,lstm_hidden_size,
                                    chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,lstm_hidden_size,
                                           chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic')

        # 初始化网络参数
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, time_step, evaluate=False):
        """
        根据当前观测值，结合策略网络输出一个动作，并添加噪声用于探索。
        :param observation: 当前观测值
        :param time_step: 当前时间步，用于计算噪声尺度
        :param evaluate: 是否为评估模式，决定是否添加噪声
        :return: 选择的动作
        """
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)  # 使用Actor网络预测动作

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
        action_np = action.detach().cpu().numpy()[0]

        # 限制动作的幅度
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np

    def update_network_parameters(self, tau=None):
        """
        使用软更新方法更新目标网络的参数。
        :param tau: 软更新系数
        """
        if tau is None:
            tau = self.tau

        # 更新目标策略网络参数
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        # 更新目标价值网络参数
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                       (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

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
