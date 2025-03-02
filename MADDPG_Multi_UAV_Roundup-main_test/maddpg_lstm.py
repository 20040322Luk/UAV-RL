import os
import torch as T
import torch.nn.functional as F
from agent import Agent


# from torch.utils.tensorboard import SummaryWriter

class LSTMMADDPG:
    """

    actor_dims 和 critic_dims 分别是每个agent的actor网络和评论家网络的输入维度。
    n_agents 是代理的数量。
    n_actions 是每个代理的动作数量。
    scenario 是场景名称，默认为 'simple'。
    alpha 和 beta 分别是演员网络和评论家网络的学习率。
    fc1 和 fc2 是演员网络和评论家网络的隐藏层大小。
    gamma 是折扣因子。
    tau 是目标网络更新的软更新参数。
    chkpt_dir 是检查点保存的目录。
    """

    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.02, fc1=128,
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []  # 声明 agent 列表
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir))
        # 初始化每个代理的演员网络和评论家网络

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, time_step, evaluate):  # timestep for exploration
        """
        根据当前观测选择动作。

        该方法主要用于在给定的时间步和评估状态下，根据每个代理的观测，为每个代理选择最合适的动作。

        参数:
        - raw_obs: 包含所有代理观测的列表。每个代理的观测可能包括环境状态、其他代理的位置等信息。
        - time_step: 当前的时间步，用于探索策略的调整。随着时间的推移，探索的程度可能会减少，以平衡探索与利用。
        - evaluate: 布尔值，指示是否为评估模式。在评估模式下，代理的行为选择可能更加保守，专注于获得最高奖励。

        返回:
        - actions: 包含所有代理所选动作的列表。每个动作是相应代理根据其观测和当前策略选择的。
        """
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):
        """
        从经验回放缓冲区中学习，更新每个智能体的策略和价值函数。

        参数:
        - memory: RNNMultiAgentReplayBuffer 对象，存储了所有智能体的时间序列经验。
        - total_steps: 总的学习步数，用于调度学习率和记录日志。

        返回值:
        无
        """
        if not memory.ready():
            return

        # 从缓冲区中采样一个批次的时间序列数据
        actor_states, states, actions, rewards_hist, states_, terminal = memory.sample_buffer()

        device = self.agents[0].actor.device

        # 转换为 PyTorch 张量并移动到设备
        states = T.tensor(states, dtype=T.float).to(device)  # 全局状态
        actions = T.tensor(actions, dtype=T.float).to(device)  # 动作序列
        rewards_hist = T.tensor(rewards_hist, dtype=T.float).to(device)  # 奖励序列
        states_ = T.tensor(states_, dtype=T.float).to(device)  # 下一全局状态
        terminal = T.tensor(terminal, dtype=T.bool).to(device)  # 终止标志

        all_agents_new_actions = []
        old_agents_actions = []

        # 遍历每个智能体，获取其新动作和旧动作
        for agent_idx, agent in enumerate(self.agents):
            # 获取当前智能体的时间序列状态
            agent_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)

            # 使用目标网络计算新动作
            with T.no_grad():
                new_pi = agent.target_actor(agent_states)  # LSTM/GRU 网络支持时间序列输入
                all_agents_new_actions.append(new_pi)

            # 提取当前智能体的旧动作
            old_agents_actions.append(actions[:, :, agent_idx * self.n_actions:(agent_idx + 1) * self.n_actions])

        # 沿着动作维度拼接所有智能体的动作
        new_actions = T.cat(all_agents_new_actions, dim=-1)  # 新动作序列
        old_actions = T.cat(old_agents_actions, dim=-1)  # 旧动作序列

        # 开始更新每个智能体的网络
        for agent_idx, agent in enumerate(self.agents):
            # 计算目标 Q 值
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards_hist[:, :, agent_idx] + (1 - terminal[:, :, 0].float()) * agent.gamma * critic_value_

            # 当前 Q 值
            critic_value = agent.critic.forward(states, old_actions).flatten()

            # Critic 损失
            critic_loss = F.mse_loss(critic_value, target)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            # Actor 损失
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:, :, agent_idx * self.n_actions:(agent_idx + 1) * self.n_actions] = agent.actor.forward(mu_states)
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # 日志记录（可选）
            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

        # 更新目标网络参数
        for agent in self.agents:
            agent.update_network_parameters()

