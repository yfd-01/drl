""" 智能体 """
import numpy as np


class Agent:
    def __init__(self, dqn, sts):
        self.forward_steps = 0
        self.total_forward_steps = 0
        self.sync_target_steps = sts
        self.dqn = dqn

    def sample(self, obs):
        """ agent根据当前状态做出动作 """
        if np.random.uniform() > self.dqn.epsilon:
            act = self.predict(obs)     # 选择最优的动作
        else:
            act = np.random.randint(self.dqn.n_actions)     # 随机选取动作

        self.dqn.epsilon = max(self.dqn.epsilon_min,
                               self.dqn.epsilon_start - self.total_forward_steps / self.dqn.epsilon_decay)

        return act

    def learn(self, states, actions, rewards, dones, states_):
        """ agent用replay buffer中mini-batch个的经验进行学习 """
        total_loss = self.dqn.learn(states, actions, rewards, dones, states_)

        if not (self.total_forward_steps % self.sync_target_steps):
            # 把dqn神经网络权重赋给target神经网络
            self.dqn.sync_target_net()
            print(f"train_loss: {total_loss}")

    def predict(self, obs):
        return self.dqn.predict(obs)
