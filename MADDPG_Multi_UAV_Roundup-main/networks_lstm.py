import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LstmNetwork(nn.Module):
    def __init__(self, input_dims, lstm_hidden_size, name, chkpt_dir):
        super(LstmNetwork, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.chkpt_file = os.path.join(chkpt_dir, name)

        # LSTM 层
        self.lstm = nn.LSTM(input_dims, lstm_hidden_size, batch_first=True)

        # 设备配置
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # 输出形状 [batch, seq_len, lstm_hidden_size]
        return lstm_out  # 返回整个时间序列的输出

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions,lstm_hidden_size, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # LSTM 输出作为输入
        self.lstm = LstmNetwork(input_dims + n_agents * n_actions, lstm_hidden_size, name + '_lstm', chkpt_dir)
        # self.fc1_dims = lstm_hidden_size
        self.fc1 = nn.Linear(lstm_hidden_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # LSTM 前向传播
        lstm_out = self.lstm(state)  # 使用 LSTM 提取时间序列特征
        x = F.relu(self.fc1(T.cat([lstm_out, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q  # 返回 Q 值和 LSTM 输出
    def save_checkpoint(self):
        """保存模型参数到文件。"""
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        """从文件加载模型参数。"""
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions,lstm_hidden_size, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # LSTM 输出作为输入
        self.lstm = LstmNetwork(input_dims, lstm_hidden_size, name + '_lstm', chkpt_dir)
        # self.fc1_dims = lstm_hidden_size
        self.fc1 = nn.Linear(lstm_hidden_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # LSTM 前向传播
        lstm_out = self.lstm(state)  # 使用 LSTM 提取时间序列特征
        x = F.leaky_relu(self.fc1(lstm_out))
        x = F.leaky_relu(self.fc2(x))
        pi = nn.Softsign()(self.pi(x))  # [-1,1]

        return pi  # 返回动作和 LSTM 输出
    def save_checkpoint(self):
        """保存模型参数到文件。"""
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        """从文件加载模型参数。"""
        self.load_state_dict(T.load(self.chkpt_file, map_location=self.device))