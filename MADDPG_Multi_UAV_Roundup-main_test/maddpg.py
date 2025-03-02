import os
import torch as T
import torch.nn.functional as F
from agent import Agent
# from torch.utils.tensorboard import SummaryWriter

class MADDPG:
    """

    actor_dims 和 critic_dims 分别是每个agent的actor网络和评论家网络的输入维度。
    n_agents 是代理的数量。
    n_actions 是每个代理的动作数量。
    scenario 是场景名称，默认为 'simple'。
    alpha 和 beta 分别是演员网络和评论家网络的学习率。
    fc1 和 fc2 是演员网络和评论家网络的隐藏层大小。
    gamma 是折扣因子。
    tau 是目标网络更新的软更新参数。
    chkpt_dir 是检查点保存的目录。
    """
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.02, fc1=128, 
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = [] # 声明 agent 列表
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))
        # 初始化每个代理的演员网络和评论家网络

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()


    def choose_action(self, raw_obs, time_step, evaluate):# timestep for exploration
        """
        根据当前观测选择动作。

        该方法主要用于在给定的时间步和评估状态下，根据每个代理的观测，为每个代理选择最合适的动作。

        参数:
        - raw_obs: 包含所有代理观测的列表。每个代理的观测可能包括环境状态、其他代理的位置等信息。
        - time_step: 当前的时间步，用于探索策略的调整。随着时间的推移，探索的程度可能会减少，以平衡探索与利用。
        - evaluate: 布尔值，指示是否为评估模式。在评估模式下，代理的行为选择可能更加保守，专注于获得最高奖励。

        返回:
        - actions: 包含所有代理所选动作的列表。每个动作是相应代理根据其观测和当前策略选择的。
        """
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx],time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):

        """
                从记忆中学习，更新代理的策略和价值函数。

                参数:
                - memory: 共享的内存缓冲区，存储了所有代理的经验。
                - total_steps: 总的学习步数，用于调度学习率和记录日志。

                返回值:
                无
        """
        if not memory.ready():
            return
        # 采样经验buffer中的数据  数据格式为 ：
        # (每个agent的st, 全局st, 每个agent的a, 每个agent的即时奖励,
        # 每个agent的st+1, 全局st+1, episode结束否)
        # actor_states, states, actions, rewards, \
        # actor_new_states, states_, dones = memory.sample_buffer()
        actor_states, states, actions, rewards, actor_new_states, states_, terminal, indices, ISWeights, dones = memory.sample_buffer()



        # --------- 采样完成 ---------

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

            new_pi = agent.target_actor.forward(new_states) # 根据新的状态使用目标网络输出新的策略

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])
        # 沿着动作维度拼接所有agent的动作
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)
        # 真正的学习开始了...
        all_td_errors = []
        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten() # 当前critic网络输出的Q值
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_ # 根据公式计算出的目标Q值

            critic_value = agent.critic.forward(states, old_actions).flatten()
            td_error = target - critic_value
            td_error = target - critic_value
            all_td_errors.append(td_error)



            critic_loss = F.mse_loss(target, critic_value) # 计算critic网络的损失
            agent.critic.optimizer.zero_grad() # 反向传播更新critic网络参数
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)  # 只更新当下索引处的智能体动作，其余的保持不变
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten()) # 计算actor网络的损失
            agent.actor.optimizer.zero_grad() # 反向传播更新actor网络参数
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
        # 合并所有智能体的 TD error
        td_error_combined = T.stack(all_td_errors, dim=0)  # 转换为张量
        # td_error_mean = T.mean(td_error_combined, dim=0)  # 按经验取平均

        # 确保 td_error_combined 是 PyTorch 张量
        if not isinstance(td_error_combined, T.Tensor):
            td_error_combined = T.tensor(td_error_combined, device=device, dtype=T.float)

        # 按经验取最大值
        td_error_max, _ = T.max(td_error_combined, dim=0)  # 只取最大值

        # 将张量转换为 NumPy 数组
        if td_error_max.dim() == 0:  # 如果是标量张量
            td_error_numpy = td_error_max.item()
        else:
            td_error_numpy = td_error_max.detach().cpu().numpy()
        # 更新优先级到经验回放缓冲区
        # td_error_numpy = td_error_max.detach().cpu().numpy()
        memory.update_priorities(indices, td_error_numpy)

        for agent in self.agents:    
            agent.update_network_parameters()

        # return indices, all_td_errors
        # return critic_loss, actor_loss