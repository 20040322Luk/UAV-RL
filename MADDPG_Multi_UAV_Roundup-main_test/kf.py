import numpy as np

class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise):
        # 时间步长
        self.dt = dt

        # 状态向量 [x, y, vx, vy]
        self.x = np.zeros(4)

        # 状态转移矩阵 F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 测量矩阵 H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 状态协方差矩阵 P
        self.P = np.eye(4)

        # 过程噪声协方差 Q
        self.Q = process_noise * np.eye(4)

        # 测量噪声协方差 R
        self.R = measurement_noise * np.eye(2)

    def predict(self):
        # 预测状态
        self.x = np.dot(self.F, self.x)
        # 预测状态协方差
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x[:2]  # 返回预测的位置 [x, y]

    def update(self, z):
        # 计算卡尔曼增益
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))

        # 更新状态向量
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(K, y)

        # 更新协方差矩阵
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
