import collections
import random
import torch


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        exps = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones, states_ = zip(*exps)

        states = torch.cat(states, dim=0).reshape((32, 1, 84, 84))
        states_ = torch.cat(states_, dim=0).reshape((32, 1, 84, 84))

        return states, torch.tensor(actions), torch.tensor(rewards), torch.tensor(dones).type(torch.IntTensor), states_


