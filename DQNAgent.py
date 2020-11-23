import numpy as np
from collections import namedtuple

import math
import random
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


'''
This code is based on the PyTorch DQN Tutorial:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''


class DQNAgent(object):

    def __init__(self, env):
        self.use_raw = False
        self.env = env
        self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
        )

        self.num_actions = len(self.env.actions)

        self.policy_net = DQN(self.env.state_shape[0],
                              self.num_actions).to(self.device)

        self.target_net = DQN(self.env.state_shape[0],
                              self.num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss = nn.SmoothL1Loss()
        self.optimizer = self.optim.SGD(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.episode = 0
        self.steps = 0

        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10

    def eval_step(self, state):
        # pick action with largest expected reward
        with torch.no_grad():
            pred = self.policy_net(state['obs'])
            # filter our invalid actions
            indices = torch.tensor(state['legal_actions'],
                                   device=self.device)
            # get rewards for valid actions
            rewards = pred.index_select(pred, 0, indices)
            # get index of max reward
            max_index = rewards.max(1)[1].item()
            # get action for that index
            action = state['legal_actions'][max_index]
            return torch.tensor([[action]],
                                device=self.device,
                                dtype=torch.long)
            # return self.policy_net(state).max(1)[1].view(1, 1)

    def step(self, state):
        sample = random.random()
        self.steps += 1

        if sample > self.eps_threshold():
            self.eval_step(state)
        else:
            return torch.tensor([[random.choice(state['legal_actions'])]],
                                device=self.device,
                                dtype=torch.long)

    def eps_threshold(self):
        return (self.eps_end + (self.eps_start - self.eps_end)
                * math.exp(-1.0 * self.steps / self.eps_decay))

    def get_state_dict(self):
        return self.policy_net.state_dict()

    def train(self, trajectory):
        self.episode += 1

        state, action, reward, next_state, done = trajectory

        self.memory.push(state, action, next_state, reward)

        self.optimize_model()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        # transpose batch
        batch = Transition(*zip(*transitions))

        self.optimizer.zero_grad()
        self.loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


'''
Example state:
    {
        'legal_actions': [1, 2, 3],
        'obs': array([1., 0., 0., 0., 1., 0., 0., 1., 0.]),
        'action_record': [[0, 'raise'], [1, 'fold']]
    }

    obs:
        0-2 = hand
        3-5 = hero chips
        6-8 = villain chips

Example action: 1

Example reward: 1.0
'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capactiy):
        self.capactiy = capactiy
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capactiy:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capactiy

    def sample(self, batch_size):
        return random.sample(self.capactiy, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(object):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(inputs, 64)
        self.output = nn.Linear(64, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
