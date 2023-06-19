import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def setSize(self, count):
        self.memory = deque(self.memory, maxlen=count)

class Agent(nn.Module):

    def __init__(self, input, output):
        super(Agent, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 16 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output)
        )
    def forward(self, x):
        return self.layer1(x)