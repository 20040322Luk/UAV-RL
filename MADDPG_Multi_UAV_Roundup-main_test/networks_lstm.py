import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    """
    输入:
    obs_history: (batch, T, obs_dim)
    action_history: (batch, T, action_dim)

    输出:
    Q值序列: (batch, T, 1)
    """
    def __init__(self, beta, lstm_input_dim, lstm_hidden_size, fc_hidden_size, n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # LSTM 层
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_size, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, 1)  # 输出 Q 值

        # 优化器和学习率调度器
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)

        # 设备配置
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs_history, action_history):
        """
        前向传播:
        - obs_history: (batch, T, obs_dim)
        - action_history: (batch, T, action_dim)
        """
        # 输入维度检查
        if len(obs_history.shape) != 3 or len(action_history.shape) != 3:
            raise ValueError("obs_history 和 action_history 的输入维度必须是 (batch, T, dim)")
        if obs_history.shape[0] != action_history.shape[0] or obs_history.shape[1] != action_history.shape[1]:
            raise ValueError("obs_history 和 action_history 的 batch 和时间步维度必须匹配")

        # 将观测和动作历史拼接
        lstm_in = T.cat([obs_history, action_history], dim=2)  # (batch, T, obs_dim + action_dim)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(lstm_in)  # (batch, T, lstm_hidden_size)

        # 全连接层
        x = F.relu(self.fc1(lstm_out))  # (batch, T, fc_hidden_size)
        q = self.fc2(x)  # (batch, T, 1)

        return q

    def save_checkpoint(self):
        """保存模型参数到文件。"""
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        """从文件加载模型参数。"""
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))


class ActorNetwork(nn.Module):
    """
    输入:
    obs_history: (batch, T, obs_dim)

    输出:
    动作序列: (batch, T, action_dim)
    """
    def __init__(self, alpha, lstm_input_dim, lstm_hidden_size, fc_hidden_size, n_actions, action_lb, action_ub, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # 动作范围
        self.action_lb = action_lb
        self.action_ub = action_ub

        # LSTM 层
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_size, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, n_actions)  # 输出动作

        # 优化器和学习率调度器
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)

        # 设备配置
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs_history):
        """
        前向传播:
        - obs_history: (batch, T, obs_dim)
        """
        # 输入维度检查
        if len(obs_history.shape) != 3:
            raise ValueError("obs_history 的输入维度必须是 (batch, T, obs_dim)")

        # LSTM 前向传播
        lstm_out, _ = self.lstm(obs_history)  # (batch, T, lstm_hidden_size)

        # 全连接层
        x = F.relu(self.fc1(lstm_out))  # (batch, T, fc_hidden_size)
        actions = self.fc2(x)  # (batch, T, n_actions)

        # 动作范围限制
        if self.action_lb is not None and self.action_ub is not None:
            mid = (self.action_lb + self.action_ub) / 2
            span = (self.action_ub - self.action_lb) / 2
            actions = span * T.tanh(actions) + mid  # 将动作限制在 [action_lb, action_ub] 范围内

        return actions

    def save_checkpoint(self):
        """保存模型参数到文件。"""
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        """从文件加载模型参数。"""
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))


