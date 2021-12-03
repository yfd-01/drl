import time

import gym
import numpy as np
import torch

from wrapper import ImageProcess, NormalizeFrame, ImageToTensor, NoopResetEnv, MaxAndSkipEnv

from model import Net
from dqn import DQN
from agent import Agent
from replayMemory import ReplayMemory

env = gym.make('Pong-v0')   # 导入交互环境
env = NoopResetEnv(env, 30)     # 增加初始画面的随机性
# env = MaxAndSkipEnv(env, 4)     # 跳帧操作
env = ImageProcess(env, 84, 84)     # 对原始画面图像进行处理，将RGB图像转换为灰度图像，再将图片裁切成 84x84
env = NormalizeFrame(env)       # 图像数据归一化操作
env = ImageToTensor(env)        # 把图像数据转换成张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 运算计算图设备

# 超参数设定
MAX_EPISODE = 600   # 训练轮数上限

MEMORY_MAX_SIZE = 10000 * 10    # 回放机制经验存储上限
MEMORY_PREPARE_SIZE = 10000     # 回放空间经验准备数量（之后开始训练DQN模型）
MEMORY_BATCH_SIZE = 32          # 训练模型的mini-batch数量

EPSILON_START = 1   # 初始的epsilon值（epsilon值与随机动作选择的概率成正相关）
EPSILON_MIN = 0.02  # 最小的epsilon值
EPSILON_DECAY = 800000
# EPSILON_DECAY = 1000000     # epsilon值的衰减程度
GAMMA = 0.99    # 未来奖励之和的折扣因子
LEARNING_RATE = 1e-4    # 梯度更新的学习率
SYNC_TARGET_STEPS = 1000    # 多少帧更新目标神经网络


net = Net(env.action_space.n).to(device)    # 预测神经网络
target_net = Net(env.action_space.n).to(device)     # 目标神经网络

# dqn算法
dqn = DQN(net, target_net, env.action_space,
          epsilon_params=(EPSILON_START, EPSILON_MIN, EPSILON_DECAY),
          gamma=GAMMA,
          lr=LEARNING_RATE)

agent = Agent(dqn, sts=SYNC_TARGET_STEPS)   # 智能体

replayMemory = ReplayMemory(MEMORY_MAX_SIZE)    # 经验回放机制


def train_episode(render=False, render_interval=0.01):
    """
    训练

    :param render: 显示画面 - `bool`
    :param render_interval: 画面刷新速度 - `float`
    :return: train_reward: 此次训练中智能体的得分总和 - `float`
    """
    train_reward = 0
    obs = env.reset()   # 重置环境

    agent.forward_steps = 0

    while True:
        agent.forward_steps += 1
        agent.total_forward_steps += 1
        action = agent.sample(obs.to(device))   # 智能体根据环境得出动作

        obs_, reward, done, info = env.step(action)     # 采取该动作与环境交互，得出 下一环境，奖励，是否结束，信息

        replayMemory.append((obs, action, reward, done, obs_))  # 把该条经验记入回放空间

        if len(replayMemory) > MEMORY_PREPARE_SIZE:
            # 从回放记录采样一个mini-batch的数据
            states, actions, rewards, dones, states_ = replayMemory.sample(MEMORY_BATCH_SIZE)
            # 智能体根据这些之前记录去学习
            agent.learn(states.to(device), actions.to(device), rewards.to(device), dones.to(device), states_.to(device))

        train_reward += reward  # 记录这一步的奖励
        obs = obs_      # 将环境状态更新

        if render:
            env.render()
            time.sleep(render_interval)

        if done:
            break

    return train_reward


def test_episode(render=True, render_interval=0.01, test_times=5):
    """
    测试

    :param render: 显示画面 - `bool`
    :param render_interval: 画面刷新速度 - `float`
    :return: mean_test_reward: 测试中智能体平均得分
    """
    test_rewards = []   # 存储测试时每一轮的奖励

    for i in range(test_times):
        obs = env.reset()
        episode_reward = 0

        while True:
            action = agent.predict(obs.to(device))  # 智能体根据环境预测动作（不是采样）
            obs, reward, done, info = env.step(action)  # 动作与环境交互
            episode_reward += reward

            if render:
                env.render()
                time.sleep(render_interval)

            if done:
                break

        test_rewards.append(episode_reward)

    return np.mean(test_rewards)    # 奖励均值返回


if __name__ == "__main__":
    pass

    # 训练
    # episode = 0
    # max_reward = -999
    #
    # # 积累回放经验中
    # while len(replayMemory) < MEMORY_PREPARE_SIZE:
    #     train_episode()
    #
    # while episode < MAX_EPISODE:
    #     train_reward = train_episode()  # 训练
    #     episode += 1
    #     print(f"Episode: {episode}   train_reward: {train_reward}   total_steps: {agent.forward_steps}")
    #
    #     # 每训练20轮，测试一次
    #     if not (episode % 20):
    #         test_reward = test_episode()    # 测试
    #         print(f"episode {episode}   e-greed {dqn.epsilon}   test_reward {test_reward}")
    #
    #         # 存储较优dqn模型
    #         if episode > MAX_EPISODE / 3 and test_reward > max_reward:
    #             max_reward = test_reward
    #             torch.save(dqn.net, f"dqn_pong_model_reward_{test_reward}_eps_{episode}")

    # 测试
    # dqn.net = torch.load("dqn_pong_model_reward_-13.0_eps_660")
    # test_episode(render=True)
