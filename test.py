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


from a2c_ppo_acktr.algo import gail


file_name = os.path.join("./gail_experts", "trajs_{}.pt".format("ant"))

expert_dataset = gail.ExpertDataset(file_name, num_trajectories=4, subsample_frequency=20)