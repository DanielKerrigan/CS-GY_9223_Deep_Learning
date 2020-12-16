import random
from collections import namedtuple

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


This code is from the PyTorch RL Tutorial

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class CircularBuffer(object):

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
