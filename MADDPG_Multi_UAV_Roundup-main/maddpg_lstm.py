import os
import torch as T
import torch.nn.functional as F
from agent_lstm import Agent
# from torch.utils.tensorboard import SummaryWriter

class LSTMDDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.02, lstm_hidden_size=64,
                 fc1=128, fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/lstmddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            lstm_hidden_size=lstm_hidden_size, chkpt_dir=chkpt_dir))

        # 初始化每个 Agent 的 LSTM 隐状态
        self.actor_hidden_states = [None] * self.n_agents
        self.critic_hidden_states = [None] * self.n_agents

    def choose_action(self, raw_obs, time_step, evaluate):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            # 更新 LSTM 隐状态
            action = agent.choose_action(
                raw_obs[agent_idx],
                time_step,
                evaluate,

            )
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                                dtype=T.float).to(device)

            # Target Actor 前向传播，传入 LSTM 隐状态
            new_pi = agent.target_actor.forward(
                new_states
            )

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(
                    states_, new_actions
                )
                critic_value_ = critic_value_.flatten()
                target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_

            critic_value, self.critic_hidden_states[agent_idx] = agent.critic.forward(
                states, old_actions, self.critic_hidden_states[agent_idx]
            )
            critic_value = critic_value.flatten()

            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)

            new_actor_actions, self.actor_hidden_states[agent_idx] = agent.actor.forward(
                mu_states, self.actor_hidden_states[agent_idx]
            )

            oa = old_actions.clone()
            oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = new_actor_actions

            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

        for agent in self.agents:
            agent.update_network_parameters()

    def reset_hidden_states(self):
        self.actor_hidden_states = [None] * self.n_agents
        self.critic_hidden_states = [None] * self.n_agents
        #
        # for agent in self.agents:
        #     agent.reset_hidden_states()

