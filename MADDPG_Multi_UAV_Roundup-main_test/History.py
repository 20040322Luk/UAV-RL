import torch

class History:
    def __init__(self, obs_dim, action_dim):
        """
        初始化历史记录模块。
        参数:
        - obs_dim: 观测空间的维度。
        - action_dim: 动作空间的维度。
        """
        self.obs_dim = obs_dim
        self.obs_hist = torch.zeros((0, obs_dim))  # 初始化观测历史为空张量

        self.action_dim = action_dim
        self.action_hist = torch.zeros((0, action_dim))  # 初始化动作历史为空张量

        self.reward_hist = torch.zeros((0, 1))  # 初始化奖励历史为空张量

    @staticmethod
    def _insert(hist, new_value):
        """
        将新的值插入到历史中。
        参数:
        - hist: 历史张量。
        - new_value: 新的值（标量或向量）。
        返回:
        - 更新后的历史张量。
        """
        # 将新值扩展维度为 (1, ...) 并转换为 float32
        new_value = torch.tensor(new_value, dtype=torch.float32).unsqueeze(0)
        return torch.cat([hist, new_value], dim=0)

    def insert_obs(self, obs):
        """
        插入新的观测值。
        参数:
        - obs: 当前时间步的观测（向量）。
        """
        self.obs_hist = self._insert(self.obs_hist, obs)

    def insert_action(self, action):
        """
        插入新的动作值。
        参数:
        - action: 当前时间步的动作（向量）。
        """
        self.action_hist = self._insert(self.action_hist, action)

    def insert_reward(self, reward):
        """
        插入新的奖励值。
        参数:
        - reward: 当前时间步的奖励（标量）。
        """
        self.reward_hist = self._insert(self.reward_hist, [reward])  # 奖励需要包装为列表以保持维度一致

    def get_action_history(self):
        """
        获取动作历史。
        返回:
        - 动作历史张量。
        """
        return self.action_hist

    def get_obs_history(self):
        """
        获取观测历史。
        返回:
        - 观测历史张量。
        """
        return self.obs_hist

    def get_reward_history(self):
        """
        获取奖励历史。
        返回:
        - 奖励历史张量。
        """
        return self.reward_hist
