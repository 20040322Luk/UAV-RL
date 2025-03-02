import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy
from scipy.optimize import linear_sum_assignment

class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_agents=4):
        self.length = length  # length of boundary
        self.num_obstacle = num_obstacle  # number of obstacles
        self.num_agents = num_agents
        self.time_step = 0.5  # update time step
        self.v_max = 0.1
        self.v_max_e = 0.12
        self.a_max = 0.04
        self.a_max_e = 0.05
        self.L_sensor = 0.2
        self.num_lasers = 16  # num of laserbeams
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0', 'agent_1', 'agent_2', 'target']
        self.info = np.random.get_state()  # get seed
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(num_agents)]
        self.last_pos = [np.zeros(2) for _ in range(num_agents)]  # 初始化为零向量

        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        }  # action represents [a_x,a_y]
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }

    def reset(self):
        SEED = random.randint(1, 1000)
        random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1:  # if not target
                self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            else:  # for target
                # self.multi_current_pos.append(np.array([1.0,0.25]))
                self.multi_current_pos.append(np.array([0.5, 1.75]))
            self.multi_current_vel.append(np.zeros(2))  # initial velocity = [0,0]
        # 初始化 last_pos 为当前的位置
        self.last_pos = [np.copy(pos) for pos in self.multi_current_pos]

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self, actions):
        # 在更新位置之前，保存当前的位置到 last_pos
        self.last_pos = [np.copy(pos) for pos in self.multi_current_pos]

        # 更新 UAV 的速度和位置
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_target = self.multi_current_pos[-1]
                # 计算当前 UAV 到目标的距离
                pass  # 如果需要，可以在这里使用 last_pos 计算上一时刻的距离

            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step  # x 坐标
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step  # y 坐标

        # 更新障碍物的位置
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones = self.cal_rewards_dones(Collided, self.last_pos)
        multi_next_obs = self.get_multi_obs()

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs

    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = []  # dim 3 only for target
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]  # dim 4
            S_team = []  # dim 4 for 3 agents 1 target
            S_target = []  # dim 2
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1] - pos[1], pos_target[0] - pos[0])
                    S_target.extend([d / np.linalg.norm(2 * self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d / np.linalg.norm(2 * self.length))

            S_obser = self.multi_current_lasers[i]  # dim 16

            if i != self.num_agents - 1:
                single_obs = [S_uavi, S_team, S_obser, S_target]
            else:
                single_obs = [S_uavi, S_obser, S_evade_d]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)

        return total_obs

    def generate_triangle_formation(self):
        """
        根据当前 UAV 的位置，动态调整生成满足以下条件的三角形队形：
        1. 每个 UAV 到质心的距离相等。
        2. 每个 UAV 之间的距离相等。
        :return: 最优的理想队形位置 (N x 2 array)
        """
        # 获取当前前三个 UAV 的位置
        current_positions = np.array(self.multi_current_pos[:3])  # 提取前三个 UAV 的位置

        # 计算当前 UAV 的质心
        centroid = np.mean(current_positions, axis=0)

        # 计算当前 UAV 到质心的距离
        distances_to_centroid = np.linalg.norm(current_positions - centroid, axis=1)

        # 使用平均距离作为目标半径，确保所有 UAV 到质心的距离相等
        target_radius = np.mean(distances_to_centroid)

        # 生成等边三角形的顶点
        # 确保每个顶点到质心的距离为 target_radius
        # 等边三角形的边长与半径的关系：边长 = sqrt(3) * 半径
        side_length = target_radius * np.sqrt(3)

        # 计算等边三角形的三个顶点（基于当前质心）
        triangle_positions = np.zeros((3, 2))
        for i in range(3):
            angle = 2 * np.pi * i / 3  # 每个顶点的角度差为 120 度
            triangle_positions[i] = [
                centroid[0] + target_radius * np.cos(angle),
                centroid[1] + target_radius * np.sin(angle)
            ]

        # 调整 UAV 到目标三角形顶点的分配
        # 计算当前 UAV 位置到目标三角形顶点的距离矩阵
        cost_matrix = np.zeros((3, 3))
        for i in range(3):  # 遍历当前 UAV
            for j in range(3):  # 遍历目标三角形顶点
                cost_matrix[i, j] = np.linalg.norm(current_positions[i] - triangle_positions[j])

        # 使用匈牙利算法解决最小化分配问题
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 根据最优分配结果调整目标位置
        optimal_positions = np.zeros_like(triangle_positions)
        for i, j in zip(row_indices, col_indices):
            optimal_positions[i] = triangle_positions[j]

        return optimal_positions

    def scale_reward(self, current_positions, desired_distance):
        """
        规模奖励 r_s
        :param current_positions: 所有无人机的当前位置 (N x 2 array)
        :param desired_distance: 理想的无人机间距离
        :return: 每个无人机的规模奖励 (1D numpy array, 长度为 num_agents - 1)
        """
        num_agents = len(current_positions)  # 获取无人机数量
        penalties = np.zeros(num_agents)  # 初始化一个大小为 num_agents 的 numpy 数组，用于存储每个无人机的惩罚值

        # 遍历前三个无人机
        for i in range(num_agents):
            penalty = 0  # 初始化当前无人机的惩罚值
            for j in range(num_agents):
                if i != j:  # 排除自身
                    # 计算当前无人机与其他无人机的距离
                    distance = np.linalg.norm(current_positions[i] - current_positions[j])
                    # 累加惩罚值
                    penalty += (desired_distance - distance) ** 2
            penalties[i] = -penalty  # 将惩罚值存储到数组中

        return penalties  # 返回每个无人机的惩罚值

    def formation_reward(self, actual_position, ideal_positions):
        """
        队形奖励 r_form
        :param actual_positions: 实际队形位置 (N x 2 array)
        :param ideal_positions: 理想队形位置 (N x 2 array)
        :return: 队形奖励
        """
        # 计算归一化因子 G(q)
        G_q = np.max(np.linalg.norm(ideal_positions[:, None] - ideal_positions, axis=2))
        # 计算队形误差
        E_p_q = np.sum(np.linalg.norm(actual_position - ideal_positions, axis=1))
        # E_p_q = self.formation_error(actual_position, ideal_positions)
        # 队形奖励
        return -E_p_q / G_q

    # def navigation_reward(self, current_distance):
    #     """
    #     导航奖励 r_navi
    #     :param prev_distance: 上一时刻到目标点的距离
    #     :param current_distance: 当前到目标点的距离
    #     :return: 导航奖励
    #     """
    #     return prev_distance - current_distance

    def obstacle_avoidance_reward(agent_positions, obstacle_positions, collision_threshold):
        """
        避障奖励 r_avoid
        :param agent_positions: 智能体位置 (N x 2 array)
        :param obstacle_positions: 障碍物位置 (M x 2 array)
        :param collision_threshold: 碰撞阈值
        :return: 避障奖励
        """
        penalty = 0
        for agent in agent_positions:
            # 计算智能体与障碍物的距离
            distances = np.linalg.norm(obstacle_positions - agent, axis=1)
            # 如果距离小于碰撞阈值，添加惩罚
            penalty += np.sum(distances < collision_threshold)
        return -penalty

    def cal_rewards_dones(self, IsCollied, last_p):
        mu1, mu2, mu3, mu4, mu5 = 0.3, 0.2, 0.6, 0.6, 0.8
        r_f = np.zeros(self.num_agents)
        r_s = np.zeros(self.num_agents)
        r_a = np.zeros(self.num_agents)
        r_o = np.zeros(self.num_agents)
        r_end = np.zeros(self.num_agents)
        d_capture = 0.6
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)

        # 获取当前无人机位置
        current_positions = np.array(self.multi_current_pos[:self.num_agents - 1])
        target_position = self.multi_current_pos[-1]
        ideal_positions = self.generate_triangle_formation()
        # for agent_i in range(self.num_agents):

        # 1奖励
        r_f[0:self.num_agents-1] += mu1 * self.formation_reward(current_positions, ideal_positions)

        # 2奖励
        l_d = 1
        r_s[0:self.num_agents-1] += mu2 * self.scale_reward(current_positions, l_d)


        # 3导航奖励
        for i in range(self.num_agents - 1):

            last_distance = np.linalg.norm(last_p[i] - target_position)
            current_distance = np.linalg.norm(self.multi_current_pos[i] - target_position)
            r_a[i] += mu3 *  100 * (last_distance - current_distance)

        # 4避障奖励 r_o (所有agent)
        # r_o = 0  # 初始化避障奖励
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10  # 碰撞惩罚
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1) / self.L_sensor
            r_o[i] += mu4 * r_safe  # 累加每个无人机的避障奖励



        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        # 4 finish rewards
        if Sum_S == S4 and all(d <= d_capture for d in [d1, d2, d3]):
            r_end[0:2] += mu5 * 30
            dones = [True] * self.num_agents


        rewards = r_f + r_s + r_a + r_o + r_end

        # 打印输出测试
        # print(f"r_f: {r_f}")
        # print(f"r_s: {r_s}")
        # print(f"r_a: {r_a}")
        # print(f"r_o: {r_o}")
        # print(f"r_end: {r_end}")



        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):

        plt.clf()

        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)

            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()

        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()

        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)

            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()


class obstacle():
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length - 0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03
        speed = 0.03  # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)

# if __name__ == '__main__':
#     # 测试编队
#     env = UAVEnv(num_obstacle=3, num_agents=4)
#     NUM_STEPS = 30
#     for i in range (NUM_STEPS):
#
#         env.reset()  # 初始化环境，生成 multi_current_pos 等属性
#
#         # 获取当前 UAV 的位置
#         current_positions = np.array(env.multi_current_pos[:3])
#
#         # 调用方法生成最优三角形队形
#         ideal_positions = env.generate_triangle_formation()
#         if i >=20:
#             # 绘图
#             plt.figure(figsize=(6, 6))
#
#             # 绘制当前 UAV 的位置
#             plt.scatter(current_positions[:, 0], current_positions[:, 1], c='blue', label='Current UAV Positions')
#             for i, pos in enumerate(current_positions):
#                 plt.text(pos[0], pos[1], f'UAV {i}', fontsize=9, color='blue')
#
#             # 绘制目标三角形队形
#             plt.scatter(ideal_positions[:, 0], ideal_positions[:, 1], c='green', label='Ideal Triangle Formation')
#             for i, pos in enumerate(ideal_positions):
#                 plt.text(pos[0], pos[1], f'Target {i}', fontsize=9, color='green')
#
#             # 连接当前 UAV 位置和目标位置
#             for curr, ideal in zip(current_positions, ideal_positions):
#                 plt.plot([curr[0], ideal[0]], [curr[1], ideal[1]], 'r--', alpha=0.6)
#
#             # 设置图形属性
#             plt.title('UAV Triangle Formation')
#             plt.xlabel('X Position')
#             plt.ylabel('Y Position')
#             plt.xlim(-0.1, env.length + 0.1)
#             plt.ylim(-0.1, env.length + 0.1)
#             plt.grid(True)
#             plt.legend()
#             plt.gca().set_aspect('equal', adjustable='box')
#
#             # 显示图形
#             plt.show()


if __name__ == '__main__':
    # 测试编队
    env = UAVEnv(num_obstacle=3, num_agents=4)
    NUM_STEPS = 30

    # 初始化环境
    multi_obs = env.reset()
    print("Initial multi_obs:", multi_obs)

    for step in range(NUM_STEPS):
        print(f"\nStep {step + 1}:")

        # 将动作的键值转换为数字索引
        actions = {
            0: np.random.uniform(-0.1, 0.1, size=(2,)),
            1: np.random.uniform(-0.1, 0.1, size=(2,)),
            2: np.random.uniform(-0.1, 0.1, size=(2,)),
            3: np.random.uniform(-0.1, 0.1, size=(2,))
        }

        # 执行一步
        multi_next_obs, rewards, dones = env.step(actions)

        # 打印上一时刻的位置
        print("Last positions (last_pos):")
        for i, pos in enumerate(env.last_pos):
            print(f"  Agent {i}: {pos}")

        # 打印当前的位置
        print("Current positions (multi_current_pos):")
        for i, pos in enumerate(env.multi_current_pos):
            print(f"  Agent {i}: {pos}")

        # 计算每个无人机到目标的距离
        target_position = env.multi_current_pos[-1]  # 目标的位置
        print("Distances to target:")
        for i in range(env.num_agents - 1):
            distance = np.linalg.norm(env.multi_current_pos[i] - target_position)
            print(f"  Agent {i} -> Target: {distance}")

        # 测试上一时刻与当前时刻的距离变化
        print("Distance changes (last_pos -> current_pos):")
        for i in range(env.num_agents - 1):
            last_distance = np.linalg.norm(env.last_pos[i] - target_position)
            current_distance = np.linalg.norm(env.multi_current_pos[i] - target_position)
            print(f"  Agent {i}: Last Distance = {last_distance}, Current Distance = {current_distance}, Change = {current_distance - last_distance}")

        # 打印奖励和终止状态
        print("Rewards:", rewards)
        print("Dones:", dones)

        # 绘制三角形队形（从第20步开始）
        if step >= 20:
            current_positions = np.array(env.multi_current_pos[:3])
            ideal_positions = env.generate_triangle_formation()

            # 绘图
            plt.figure(figsize=(6, 6))

            # 绘制当前 UAV 的位置
            plt.scatter(current_positions[:, 0], current_positions[:, 1], c='blue', label='Current UAV Positions')
            for i, pos in enumerate(current_positions):
                plt.text(pos[0], pos[1], f'UAV {i}', fontsize=9, color='blue')

            # 绘制目标三角形队形
            plt.scatter(ideal_positions[:, 0], ideal_positions[:, 1], c='green', label='Ideal Triangle Formation')
            for i, pos in enumerate(ideal_positions):
                plt.text(pos[0], pos[1], f'Target {i}', fontsize=9, color='green')

            # 连接当前 UAV 位置和目标位置
            for curr, ideal in zip(current_positions, ideal_positions):
                plt.plot([curr[0], ideal[0]], [curr[1], ideal[1]], 'r--', alpha=0.6)

            # 设置图形属性
            plt.title(f'UAV Triangle Formation (Step {step + 1})')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.xlim(-0.1, env.length + 0.1)
            plt.ylim(-0.1, env.length + 0.1)
            plt.grid(True)
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')

            # 显示图形
            plt.show()


