import numpy as np


class SumTree:
    """
    SumTree 数据结构，用于存储优先级和支持快速采样。
    """
    def __init__(self, capacity):
        self.capacity = capacity  # SumTree 的容量（叶子节点数量）
        self.tree = np.zeros(2 * capacity - 1)  # 二叉树数组表示
        self.data = [None] * capacity  # 存储实际经验
        self.data_pointer = 0  # 当前存储位置指针

    def add(self, p, data):
        """
        添加新经验及其优先级。
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # 存储经验
        self.update(tree_idx, p)  # 更新优先级

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # 循环覆盖最旧的数据

    def update(self, tree_idx, p):
        """
        更新指定叶子节点的优先级，并向上传递变化。
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:  # 向上传递变化
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        根据随机值 v 采样叶子节点。
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # 左子节点索引
            cr_idx = cl_idx + 1  # 右子节点索引
            if cl_idx >= len(self.tree):  # 到达叶子节点
                leaf_idx = parent_idx
                break
            else:  # 向下搜索
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        """
        返回所有优先级的总和（根节点值）。
        """
        return self.tree[0]


class PERMultiAgentReplayBuffer:
    """
    带有优先经验回放的多智能体经验缓冲区
    """
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size, alpha=0.7, beta=0.3, abs_err_upper=10):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.alpha = alpha  # 优先级的敏感度
        self.beta = beta  # 用于重要性采样权重
        self.abs_err_upper = abs_err_upper  # 优先级的上限

        # 使用 SumTree 存储优先级和经验
        self.sum_tree = SumTree(max_size)

        # 初始化经验存储
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []  # 当前状态记忆
        self.actor_new_state_memory = []  # 下一状态记忆
        self.actor_action_memory = []  # 动作记忆

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        """
        存储多智能体的经验
        """
        index = self.mem_cntr % self.mem_size

        # 存储多智能体的状态、动作、奖励等
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # 为新经验分配一个较大的初始优先级
        max_p = np.max(self.sum_tree.tree[-self.mem_size:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sum_tree.add(max_p, index)

        self.mem_cntr += 1

    def sample_buffer(self):
        """
        根据优先级采样经验
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        segment = self.sum_tree.total_p / self.batch_size  # 总优先级分段
        indices = []
        priorities = []
        ISWeights = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, data_idx = self.sum_tree.get_leaf(v)
            indices.append(data_idx)
            priorities.append(priority)

        # 计算重要性采样权重
        total_p = self.sum_tree.total_p
        min_p = np.min(self.sum_tree.tree[-max_mem:]) / total_p
        for i, p in enumerate(priorities):
            ISWeights[i] = (p / total_p) ** -self.beta
        ISWeights /= ISWeights.max()  # 归一化

        # 提取对应的经验
        batch = np.array(indices)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal, indices, ISWeights, terminal

    def update_priorities(self, indices, td_errors):
        """
        根据 TD-误差更新优先级
        """
        for i, idx in enumerate(indices):
            p = np.abs(td_errors[i]) + 1e-5  # 避免优先级为 0
            p = np.clip(p, 0, self.abs_err_upper)
            p = p ** self.alpha
            self.sum_tree.update(idx + self.mem_size - 1, p)

    def ready(self):
        return self.mem_cntr >= self.batch_size
