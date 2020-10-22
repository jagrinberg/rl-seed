import copy
import glob
import os
import time
from collections import deque

import pybulletgym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize
from evaluation import evaluate


def main():
    #Get arguments from command line
    args = get_args()

    #Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #Helps with reproducibility?
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #Where to log
    if args.i != None:
        args.log_dir += str(args.i)
    args.log_dir += "-" + str(args.seed)
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    #Setup torch
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
   
    
    # envs_gail = make_vec_envs(args.env_name, args.seed, 1,
                         # args.gamma, '/tmp/gym/', device, False)

    envs_seed = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    
    aenv_name = args.env_name
    
    # genv_name = "InvertedPendulumSwingupPyBulletEnv-v0"
    genv_name = args.env_name
    
    # args.env_name = "Sub" + args.env_name
    
    if args.train_gail:
        init_gail = 29
    else:
        init_gail = -1
    update_freq = 5
    
    #Create base policy
    actor_critic_g = Policy(
        envs_seed.observation_space.shape,
        envs_seed.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic_g.to(device)

    actor_critic_s = Policy(
        envs_seed.observation_space.shape,
        envs_seed.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic_s.to(device)
    
    
    
    if not(args.train_gail):
        actor_critic_g, ob_rms = \
                torch.load(os.path.join(args.gail_agent_dir, genv_name + "gail.pt"))

        # vec_norm = get_vec_normalize(envs_gail)
        # vec_norm2 = get_vec_normalize(envs_seed)
        # if vec_norm2 is not None:

            # vec_norm2.eval()

            # vec_norm2.obs_rms = ob_rms

    #Set algorithm based on input
    gailfilt = lambda x: np.clip((x - ob_rms.mean) / np.sqrt(ob_rms.var + utils.get_vec_normalize(envs_seed).epsilon), -utils.get_vec_normalize(envs_seed).clip_obs, utils.get_vec_normalize(envs_seed).clip_obs)
    
    if args.algo == 'a2c':
        agent_g = algo.A2C_ACKTR(
            actor_critic_g,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
        agent_s = algo.A2C_ACKTR(
            actor_critic_s,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent_g = algo.PPO(
            actor_critic_g,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        agent_s = algo.SEED(
            actor_critic_s,
            actor_critic_g,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.kl_coef,
            lr=args.lr,
            eps=args.eps,
            steps=args.num_steps*args.num_processes,
            env_steps=args.num_env_steps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent_g = algo.A2C_ACKTR(
            actor_critic_g, args.value_loss_coef, args.entropy_coef, acktr=True)
        agent_s = algo.A2C_ACKTR(
            actor_critic_s, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs_seed.observation_space.shape) == 1
        #Create discriminator
        acti = 0
        space = None
        if envs_seed.action_space.__class__.__name__ == "Discrete":
            acti = envs_seed.action_space.n
            space = acti
        else:
            acti = envs_seed.action_space.shape[0]
        
        print(acti)
        if args.train_gail:
            discr = gail.Discriminator(
                envs_seed.observation_space.shape[0] + acti, 100,
                device, space)
        else:
            discr = torch.load(os.path.join(args.gail_agent_dir, genv_name + "discr.pt"))[0]
            print(discr)
        #Get filename based on environment
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                "mountaincarcontinuous"))
        #Store dataset
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=2)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    
    # rollouts_g = RolloutStorage(args.num_steps, args.num_processes,
                              # envs_gail.observation_space.shape, envs_gail.action_space,
                              # actor_critic_g.recurrent_hidden_state_size)
    rollouts_s = RolloutStorage(args.num_steps, args.num_processes, 
                                envs_seed.observation_space.shape, envs_seed.action_space, 
                                actor_critic_s.recurrent_hidden_state_size)
    
    orig_obs = torch.zeros(args.num_steps + 1, args.num_processes, *envs_seed.observation_space.shape)
    
    # obs_gail = envs_gail.reset()
    # rollouts_g.obs[0].copy_(obs_gail)
    # rollouts_g.to(device)
    
        
    
    obs_seed = envs_seed.reset()
    orig_obs[0].copy_(torch.from_numpy(utils.get_vec_normalize(envs_seed).get_original_obs()))
    rollouts_s.obs[0].copy_(obs_seed)
    rollouts_s.to(device)

    episode_rewards_g = deque(maxlen=10)
    episode_rewards_s = deque(maxlen=10)
    
    episode_rewards_s.append(0)
    episode_rewards_g.append(0)
    
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        # vec_norm_g = get_vec_normalize(envs_gail)
        # vec_norm_s = get_vec_normalize(envs_seed)
        # if vec_norm_s is not None:
            # vec_norm_s.eval()
            # vec_norm_s.obs_rms = ob_rms
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent_g.optimizer, j, num_updates,
                agent_g.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(
                agent_s.optimizer, j, num_updates,
                agent_s.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                # if (j < init_gail or j%update_freq==0) and args.train_gail:
                    # value_g, action_g, action_log_prob_g, recurrent_hidden_states_g = actor_critic_g.act(
                        # rollouts_g.obs[step], rollouts_g.recurrent_hidden_states[step],
                        # rollouts_g.masks[step])
                if j > init_gail:
                    value_s, action_s, action_log_prob_s, recurrent_hidden_states_s = actor_critic_s.act(
                        rollouts_s.obs[step], rollouts_s.recurrent_hidden_states[step],
                        rollouts_s.masks[step])
            # Obser reward and next obs
            # if (j < init_gail or j%update_freq==0) and args.train_gail:
                # obs_g, reward_g, done_g, infos_g = envs_gail.step(torch.squeeze(action_g))
                # obs_g, reward_g, done_g, infos_g = envs_gail.step(action_g)
                
                # for info in infos_g:
                    # if 'episode' in info.keys():
                        # episode_rewards_g.append(info['episode']['r'])

                # If done then clean the history of observations.
                # masks_g = torch.FloatTensor(
                    # [[0.0] if done_ else [1.0] for done_ in done_g])
                # bad_masks_g = torch.FloatTensor(
                    # [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     # for info in infos_g])
                
                # rollouts_g.insert(obs_g, recurrent_hidden_states_g, action_g,
                                # action_log_prob_g, value_g, reward_g, masks_g, bad_masks_g)
            
            if j > init_gail:
                # obs_s, reward_s, done_s, infos_s = envs_seed.step(torch.squeeze(action_s))
                obs_s, reward_s, done_s, infos_s = envs_seed.step(action_s)
                
                for info in infos_s:
                    if 'episode' in info.keys():
                        episode_rewards_s.append(info['episode']['r'])
                masks_s = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_s])
                bad_masks_s = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos_s])
                rollouts_s.insert(obs_s, recurrent_hidden_states_s, action_s,
                            action_log_prob_s, value_s, reward_s, masks_s, bad_masks_s)
                orig_obs[step+1].copy_(torch.from_numpy(utils.get_vec_normalize(envs_seed).get_original_obs()))

            
            


        with torch.no_grad():
            if (j < init_gail or j%update_freq==0) and args.train_gail:
                next_value_g = actor_critic_g.get_value(
                    rollouts_g.obs[-1], rollouts_g.recurrent_hidden_states[-1],
                    rollouts_g.masks[-1]).detach()
            if j > init_gail:
                next_value_s = actor_critic_s.get_value(
                    rollouts_s.obs[-1], rollouts_s.recurrent_hidden_states[-1],
                    rollouts_s.masks[-1]).detach()
        
        total = 0
        if args.gail:
            # if j >= 10:
                # envs_gail.venv.eval()
            # gail_epoch = args.gail_epoch
            # if j < 10:
                # gail_epoch = 100  # Warm up
            # if (j < init_gail or j%update_freq==0) and args.train_gail:
                # for _ in range(gail_epoch):
                    # discr.update(gail_train_loader, rollouts_g,
                                 # utils.get_vec_normalize(envs_gail)._obfilt)

            for step in range(args.num_steps):
                if (j < init_gail or j%update_freq==0) and args.train_gail:
                    rollouts_g.rewards[step], h = discr.predict_reward(
                        rollouts_g.obs[step], rollouts_g.actions[step], args.gamma,
                        rollouts_g.masks[step])
                    total += h
                if j > init_gail:
                    if j < num_updates/2:
                        r_s, h = discr.predict_reward_seed(
                            orig_obs[step], rollouts_s.actions[step], args.gamma,
                            rollouts_s.masks[step], gailfilt)
                        r_s = torch.clamp(r_s, -5, 5)
                    
                        rollouts_s.rewards[step]= -args.demonstration_coef*r_s*2*(.5-j/num_updates)+rollouts_s.rewards[step]
                     
                    else:
                        rollouts_s.rewards[step] = rollouts_s.rewards[step]
            total = total/args.num_steps/args.num_processes
        
        if (j < init_gail or j%update_freq==0) and args.train_gail:
            rollouts_g.compute_returns(next_value_g, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)
        if j > init_gail:
            rollouts_s.compute_returns(next_value_s, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)
        
        # if j % 2 == 0:
            # value_loss_g, action_loss_g, dist_entropy_g = agent_g.update(rollouts_g)
            # rollouts_g.after_update()
        # else:
            # value_loss_s, action_loss_s, dist_entropy_s = agent_s.update(rollouts_s)
            # rollouts_s.after_update()
        if (j < init_gail or j%update_freq==0) and args.train_gail:
            value_loss_g, action_loss_g, dist_entropy_g = agent_g.update(rollouts_g)
            rollouts_g.after_update()
        
        if j > init_gail:
            value_loss_s, action_loss_s, dist_entropy_s = agent_s.update(rollouts_s, orig_obs, gailfilt)
            rollouts_s.after_update()
            orig_obs[0].copy_(orig_obs[-1])
   
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            env_name = args.env_name.split(":")[-1]
            # torch.save([
                # actor_critic_g,
                # getattr(get_vec_normalize(envs_gail), 'obs_rms', None)
            # ], os.path.join(save_path, env_name + "gail.pt"))
            it = ""
            if args.i != None:
                it = args.i
            torch.save([
                actor_critic_s,
                getattr(get_vec_normalize(envs_seed), 'obs_rms', None)
            ], os.path.join(save_path, env_name + str(it) + "seed.pt"))

        if j % args.log_interval == 0 and len(episode_rewards_s) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            # print("GAIL")
            # print(total)
            # print(
                # "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n"
                # .format(j, total_num_steps,
                        # int(total_num_steps / (end - start)),
                        # len(episode_rewards_g), np.mean(episode_rewards_g),
                        # np.median(episode_rewards_g), np.min(episode_rewards_g),
                        # np.max(episode_rewards_g)))
            if j > init_gail:
                print(episode_rewards_s)
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards_s), np.mean(episode_rewards_s),
                            np.median(episode_rewards_s), np.min(episode_rewards_s),
                            np.max(episode_rewards_s)))
        num = args.num_processes
        if (args.eval_interval is not None and len(episode_rewards_s) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs_seed).obs_rms
            evaluate(actor_critic_s, obs_rms, aenv_name, args.seed,
                     num, eval_log_dir, device)

if __name__ == "__main__":
    main()
