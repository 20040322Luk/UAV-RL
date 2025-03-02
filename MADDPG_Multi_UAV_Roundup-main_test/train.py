import numpy as np
from maddpg_lstm import LSTMMADDPG
from sim_env import UAVEnv
from RNNbuffer import RNNMultiAgentReplayBuffer
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

    # 动作空间假设每个智能体有相同的动作空间
    n_actions = 2
    maddpg_agents = LSTMMADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=256, fc2=256,
                           alpha=0.0001, beta=0.003, scenario='UAV_Round_up')


    # 初始化经验回放缓冲区，增加seq_len参数
    seq_len = 10  # 时间序列长度
    memory = RNNMultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=128, seq_len=seq_len)

    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -30

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False] * n_agents
        episode_step = 0
        while not any(dones):
            if evaluate:
                # 渲染环境
                env_render = env.render()
                if episode_step % 10 == 0:
                    # 每10步保存一次图像
                    filename = f'images/episode_{i}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 创建目录
                    save_image(env_render, filename)

            # 获取动作
            actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
            obs_, rewards, dones = env.step(actions)

            # 将观测转换为全局状态
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                dones = [True] * n_agents

            # 存储时间步数据到缓冲区
            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            # 当缓冲区准备好时，采样数据并训练
            if total_steps % 10 == 0 and not evaluate and memory.ready():
                actor_states, states, actions, rewards_hist, states_, terminal = memory.sample_buffer()
                maddpg_agents.learn(memory, total_steps)

            obs = obs_
            score += sum(rewards[0:2])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score', avg_score, '>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score),
                  '; average target score {:.1f}'.format(avg_target_score))

    # 保存分数历史记录
    file_name = 'score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)
