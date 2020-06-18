import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



env = gym.make("CartPole-v1")
print(env.action_space.n)
a = np.zeros((2,3,1))
a[1][2][0] = 2
b = np.ones((2,4,1))
m = np.zeros((a.shape[0],a.shape[1],4))
a = torch.from_numpy(a)
b = torch.from_numpy(b)
for x in range(a.shape[0]):
    for y in range(a.shape[1]):
        m[x][y][a[x][y][0].long()]=1
a = torch.from_numpy(m)
print(a)