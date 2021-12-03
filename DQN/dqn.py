""" 实现DQN的核心算法 """
import torch
from torch import nn, optim


class DQN:
    def __init__(self, net, target_net, action_spaces, epsilon_params, gamma, lr):
        self.net = net
        self.target_net = target_net

        self.action_spaces = action_spaces
        self.n_actions = action_spaces.n
        self.epsilon_start = epsilon_params[0]
        self.epsilon_min = epsilon_params[1]
        self.epsilon_decay = epsilon_params[2]
        self.epsilon = self.epsilon_start
        self.gamma = gamma
        self.lr = lr

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def sync_target_net(self):
        """ 权重同步 """
        self.target_net.load_state_dict(self.net.state_dict())

    def predict(self, obs):
        """ 选取动作价值函数中最优动作 """
        return self.net(obs).detach().argmax().item()

    def learn(self, states, actions, rewards, dones, states_):
        """ 训练dqn神经网络 """
        total_loss = 0.0

        # 计算TD target
        with torch.no_grad():
            target_values = self.target_net(states_)    # 下一状态在target神经网络中的折扣回报
            target_max_values = target_values.max(dim=1)[0]     # 选取最高值
            targets = rewards + (1 - dones) * self.gamma * target_max_values
            targets = targets.reshape(32, 1)

        # 计算dqn神经网络预测值
        values = self.net(states).gather(1, actions.view(-1, 1))

        self.optimizer.zero_grad()

        loss = self.criterion(values, targets)
        loss.backward()

        # 更新dqn神经网络权值w
        self.optimizer.step()

        total_loss += loss.item()

        return total_loss
