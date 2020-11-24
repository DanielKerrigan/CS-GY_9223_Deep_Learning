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

    def __init__(self, num_actions, state_shape):
        self.use_raw = False
        self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
        )

        self.num_actions = num_actions

        self.policy_net = DQN(state_shape, num_actions).to(self.device)

        self.target_net = DQN(state_shape, num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)

        self.train_step = 0
        self.action_chosen_in_training = 0
        self.weight_updates = 0

        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10

    def choose_action(self, state):
        # pick action with largest expected reward
        with torch.no_grad():
            model_input = torch.tensor(state['obs'],
                                       dtype=torch.float,
                                       device=self.device)
            pred = self.policy_net(model_input)
            # print(f'pred = {pred}')
            # filter our invalid actions
            indices = torch.tensor(state['legal_actions'],
                                   dtype=torch.long,
                                   device=self.device)
            # print(f'indices = {indices}')
            # get rewards for valid actions
            rewards = pred.gather(0, indices)
            # print(f'rewards = {rewards}')
            # get index of max reward
            max_index = rewards.max(0)[1].item()
            # print(f'max_index = {max_index}')
            # get action for that index
            action = state['legal_actions'][max_index]
            # print(f'action = {action}')
            # print(f'legal = {state["legal_actions"]}')
            return action
            # return torch.tensor([[action]],
            #                    dtype=torch.long,
            #                    device=self.device)

    def step(self, state):
        # print('in dqn step')
        sample = random.random()
        self.action_chosen_in_training += 1

        if sample > self.eps_threshold():
            # print('dqn model')
            return self.choose_action(state)
        else:
            # print('dqn random')
            return random.choice(state['legal_actions'])
            '''
            return torch.tensor([[random.choice(state['legal_actions'])]],
                                device=self.device,
                                dtype=torch.long))
            '''

    def eval_step(self, state):
        self.policy_net.eval()
        result = (self.choose_action(state), None)
        self.policy_net.train()
        return result

    def eps_threshold(self):
        return (self.eps_end + (self.eps_start - self.eps_end)
                * math.exp(-1.0 * self.train_step / self.eps_decay))

    def get_state_dict(self):
        return self.policy_net.state_dict()

    def train(self, trajectory):
        self.train_step += 1

        self.memory.push(*trajectory)

        self.optimize_model()

    def pad_actions(self, act):
        return act + ([act[0]] * (self.num_actions - len(act)))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        # transpose the batch so that we get a transition
        # of batch array
        batch = Transition(*zip(*transitions))

        # mask for non-final states
        # these are states where we will have another move to make
        # after the current move
        non_final_mask = torch.logical_not(torch.tensor(
                batch.done,
                dtype=torch.bool,
                device=self.device
        ))

        # get the next states that are not final
        non_final_next_state_batch = torch.tensor(
                [s['obs'] for s in batch.next_state],
                dtype=torch.float,
                device=self.device
        )[non_final_mask]

        # get the legal actions for these non-final
        # next states
        non_final_next_state_legal_actions = torch.stack(
                [
                    torch.tensor(
                        self.pad_actions(s['legal_actions']),
                        dtype=torch.long,
                        device=self.device)
                    for s in batch.next_state
                ],
                dim=0
        )[non_final_mask]

        # get the state, action, and reward batches

        state_batch = torch.tensor(
                [s['obs'] for s in batch.state],
                dtype=torch.float,
                device=self.device
        )

        action_batch = torch.tensor(
                batch.action,
                dtype=torch.long,
                device=self.device
        ).view(-1, 1)

        reward_batch = torch.tensor(
                batch.reward,
                dtype=torch.float,
                device=self.device
        )

        # compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1,
                                                                  action_batch)
        # compute max a for Q(s_{t+1}, a)

        # if s_{t+1} is a final state, then Q(s_{t+1}, a) is 0
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        if non_final_next_state_batch.size()[0] != 0:
            # get predicted rewards for all non-final next states
            next_state_all_values = self.target_net(non_final_next_state_batch)

            # only select rewards for valid actions
            next_state_valid_values = next_state_all_values.gather(
                    1,
                    non_final_next_state_legal_actions
            )
            # get the max reward for a valid action
            next_state_values[non_final_mask] = (
                    next_state_valid_values.max(1)[0]
            )

        expected_state_action_values = (
                (next_state_values * self.gamma) + reward_batch
        )

        # minimize Q(s_t, a) - reward + (gamma * max a Q(s_{t+1}, a))
        # the predicted total reward for choosing action a at state s_t
        # should equal the actual reward for that action plus the
        # predicted total reward for choosing another action at state s_{t+1}
        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        self.weight_updates += 1

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.weight_updates % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self, path):
        state_dict = torch.load(path)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.load_state_dict(state_dict)


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
                        ('state', 'action', 'reward', 'next_state', 'done'))


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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(inputs, 64)
        self.output = nn.Linear(64, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
