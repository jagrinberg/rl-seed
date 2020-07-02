import argparse
import os
# workaround to unpickle olf model files
import sys

import pygame

import gym
from gym.utils import play

import numpy as np
import torch

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--num-traj',
    type=int,
    default=10,
    help='number of trajectories to save (default: 10')
args = parser.parse_args()



dic = {
    tuple([pygame.K_a]): 0,
    tuple([pygame.K_d]): 1
}

class Human():
    def __init__(self, env):
        self.count = 0
        self.max_t = 200
        self.rewards = np.zeros((args.num_traj,self.max_t))
        self.lengths = np.zeros((args.num_traj))
        self.states = np.zeros((args.num_traj,self.max_t,env.observation_space.shape[0]))
        self.actions = np.zeros((args.num_traj,self.max_t,env.action_space.n))
        self.col = 0
        
    
    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        print(action)
        self.states[self.count][self.col] = obs_t
        
        self.rewards[self.count][self.col] = rew
        self.actions[self.count][self.col][action] = 1

        self.col+=1
        if done:
            self.lengths[self.count]=self.col
            self.col = 0
            self.count += 1
            print(done)

    def save(self):
        states = torch.from_numpy(self.states).float()
        actions = torch.from_numpy(self.actions).float()
        rewards = torch.from_numpy(self.rewards).float()
        lens = torch.from_numpy(self.lengths).long()

        data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'lengths': lens
            }

        path = "trajs_{}.pt".format(args.env_name.split('-')[0].lower())
        torch.save(data, path)

env = gym.make(args.env_name)

hum = Human(env)

play.play(env, fps=30, callback=hum.callback, keys_to_action=dic)