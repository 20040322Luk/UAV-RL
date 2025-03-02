import torch
import numpy as np
from  History import History

class RNNMultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size, seq_len):
        """
        初始化多智能体经验回放缓冲区。
        参数:
        - max_size: 缓冲区最大容量。
        - critic_dims: 全局状态的维度。
        - actor_dims: 每个智能体的局部状态维度列表。
        - n_actions: 动作维度。
        - n_agents: 智能体数量。
        - batch_size: 批次大小。
        - seq_len: 时间序列长度。
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.seq_len = seq_len  # 时间序列长度

        # 全局状态和奖励存储
        self.state_memory = np.zeros((self.mem_size, seq_len, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, seq_len, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, seq_len, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, seq_len, n_agents), dtype=bool)

        # 初始化每个智能体的历史存储
        self.agent_histories = [
            History(obs_dim=actor_dims[i], action_dim=n_actions) for i in range(self.n_agents)
        ]

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        """
        存储单步数据到历史，并在历史长度达到 seq_len 时存入缓冲区。
        参数:
        - raw_obs: 每个智能体的局部状态（列表，长度为 n_agents）。
        - state: 全局状态。
        - action: 每个智能体的动作（列表，长度为 n_agents）。
        - reward: 每个智能体的奖励（列表，长度为 n_agents）。
        - raw_obs_: 每个智能体的下一局部状态（列表，长度为 n_agents）。
        - state_: 下一全局状态。
        - done: 每个智能体的终止标志（列表，长度为 n_agents）。
        """
        # 将每个智能体的观测、动作和奖励插入历史
        for agent_idx in range(self.n_agents):
            self.agent_histories[agent_idx].insert_obs(raw_obs[agent_idx])
            self.agent_histories[agent_idx].insert_action(action[agent_idx])
            self.agent_histories[agent_idx].insert_reward(reward[agent_idx])

        # 如果历史长度达到 seq_len，则将历史存入缓冲区
        if self.agent_histories[0].get_obs_history().shape[0] >= self.seq_len:
            index = self.mem_cntr % self.mem_size  # 缓冲区索引

            # 提取每个智能体的历史观测并拼接成全局状态
            agent_obs_histories = [
                self.agent_histories[agent_idx].get_obs_history() for agent_idx in range(self.n_agents)
            ]
            global_state = torch.cat(agent_obs_histories, dim=-1)  # 拼接沿最后一维（特征维度）

            # 确保 global_state 的形状与 state_memory 的单步存储形状匹配
            if global_state.shape[1] != self.state_memory.shape[2]:
                raise ValueError(f"Global state dimensions ({global_state.shape[1]}) do not match "
                                 f"critic_dims ({self.state_memory.shape[2]}).")

            # 存储到缓冲区
            self.state_memory[index] = global_state.numpy()
            self.new_state_memory[index] = state_
            self.reward_memory[index] = np.array(reward)
            self.terminal_memory[index] = np.array(done)

            # 清除每个智能体的历史记录，保留最近 seq_len - 1 的数据
            for agent_idx in range(self.n_agents):
                self.agent_histories[agent_idx].obs_hist = self.agent_histories[agent_idx].obs_hist[
                                                           -(self.seq_len - 1):]
                self.agent_histories[agent_idx].action_hist = self.agent_histories[agent_idx].action_hist[
                                                              -(self.seq_len - 1):]
                self.agent_histories[agent_idx].reward_hist = self.agent_histories[agent_idx].reward_hist[
                                                              -(self.seq_len - 1):]

            self.mem_cntr += 1

    def sample_buffer(self):
        """
        从缓冲区中采样一个批次的时间序列数据。
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # 当前有效存储数量
        batch = np.random.choice(max_mem, self.batch_size, replace=False)  # 随机采样序列索引

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actions = []
        rewards_hist = []

        # 提取每个智能体的历史记录
        for agent_idx in range(self.n_agents):
            obs_hist = torch.stack(
                [self.agent_histories[agent_idx].get_obs_history() for _ in batch]
            )
            action_hist = torch.stack(
                [self.agent_histories[agent_idx].get_action_history() for _ in batch]
            )
            reward_hist = torch.stack(
                [self.agent_histories[agent_idx].get_reward_history() for _ in batch]
            )
            actor_states.append(obs_hist)
            actions.append(action_hist)
            rewards_hist.append(reward_hist)

        return actor_states, states, actions, rewards_hist, states_, terminal

    def ready(self):
        """
        检查缓冲区是否有足够的数据进行采样。
        """
        return self.mem_cntr >= self.batch_size
