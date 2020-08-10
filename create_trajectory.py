import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch
import pybulletgym

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

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

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cuda',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)
# render_func = None

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

count = 0

max_t = 1000
rewards = np.zeros((args.num_traj,max_t))
lengths = np.zeros((args.num_traj))
states = np.zeros((args.num_traj,max_t,env.observation_space.shape[0]))

col = 0

cont = False

if env.action_space.__class__.__name__ != "Discrete":
    cont = True
    actions = np.zeros((args.num_traj,max_t,env.action_space.shape[0]))
else:
    actions = np.zeros((args.num_traj,max_t,env.action_space.n))
    
while count < args.num_traj:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)
    # Obser reward and next obs
    states[count][col] = (get_vec_normalize(env).get_original_obs())
    obs, reward, done, _ = env.step(action)
    rewards[count][col] = reward
    if cont:
        actions[count][col] = action.cpu().numpy()
    else:
        actions[count][col][action.squeeze()] = 1
    masks.fill_(0.0 if done else 1.0)
    col+=1
    if done:
        lengths[count]=col
        col = 0
        count += 1
        print(done)
    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
states = torch.from_numpy(states).float()
actions = torch.from_numpy(actions).float()
rewards = torch.from_numpy(rewards).float()
lens = torch.from_numpy(lengths).long()

data = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'lengths': lens
    }

path = "gail_experts/trajs_{}.pt".format(args.env_name.split('-')[0].lower())
torch.save(data, path)