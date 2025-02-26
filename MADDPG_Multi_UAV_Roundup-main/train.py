import numpy as np
from maddpg_lstm import LSTMDDPG  # 假设你已经实现了 LSTM-DDPG 类
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state


def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency

    image.save(filename)


if __name__ == '__main__':

    env = UAVEnv()
    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 2
    lstm_ddpg_agents = LSTMDDPG(actor_dims, critic_dims, n_agents, n_actions,
                                fc1=128, fc2=128,  # LSTM 相关参数
                                alpha=0.0001, beta=0.003, scenario='UAV_Round_up')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=256)

    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -30

    if evaluate:
        # lstm_ddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False] * n_agents
        episode_step = 0

        # 初始化 LSTM 隐状态
        lstm_ddpg_agents.reset_hidden_states()

        while not any(dones):
            if evaluate:
                # env.render()
                env_render = env.render()
                if episode_step % 10 == 0:
                    # Save the image every 10 episode steps
                    filename = f'images/episode_{i}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create directory if it doesn't exist
                    save_image(env_render, filename)
                # time.sleep(0.01)

            # 获取动作，同时更新 LSTM 隐状态
            actions = lstm_ddpg_agents.choose_action(obs, total_steps, evaluate)

            # 环境交互
            obs_, rewards, dones = env.step(actions)

            # 状态向量化
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            # 达到最大步数时终止
            if episode_step >= MAX_STEPS:
                dones = [True] * n_agents

            # 存储经验
            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            # 训练网络
            if total_steps % 10 == 0 and not evaluate:
                lstm_ddpg_agents.learn(memory, total_steps)

            # 更新观测和累计奖励
            obs = obs_
            score += sum(rewards[0:2])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1

        # 记录分数
        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])

        # 保存模型
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score', avg_score, '>', best_score, 'saving models...')
                lstm_ddpg_agents.save_checkpoint()
                best_score = avg_score

        # 打印进度
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score),
                  '; average target score {:.1f}'.format(avg_target_score))

    # 保存分数历史
    file_name = 'score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)
