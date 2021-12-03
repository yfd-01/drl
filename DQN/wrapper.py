import gym
import cv2
import numpy as np
from torch import from_numpy


class ImageProcess(gym.ObservationWrapper):
    """ 对原始画面图像进行处理，将RGB图像转换为灰度图像，再将图片裁切成 84x84 """
    def __init__(self, env, width, height, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale

    def observation(self, observation):
        frame = observation
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame[34: 194], (self.width, self.height), interpolation=cv2.INTER_AREA)
            frame = np.expand_dims(frame, -1)

        return frame


class NormalizeFrame(gym.ObservationWrapper):
    """ 图像数据归一化操作 """
    def observation(self, observation):
        return (np.array(observation).astype(np.float32)) / 255.0


class ImageToTensor(gym.ObservationWrapper):
    """ 把图像数据转换成张量 """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return from_numpy(np.moveaxis(observation, -1, 0))


class RewardScaler(gym.RewardWrapper):
    """ 同比缩小奖励 """
    def reward(self, reward):
        return reward * 0.1


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
