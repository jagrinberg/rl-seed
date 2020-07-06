import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class SEED():
    def __init__(self,
                 actor_critic,
                 gail_learner,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 kl_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic
        self.kl_coef = kl_coef
        
        
        self.log_kl = torch.zeros(
                1, requires_grad=True, device='cuda')
        
        self.target_kl = .1
        
        self.kl_coef = self.log_kl.exp()/10
        self.kl_optim = optim.Adam([self.log_kl], lr=lr)
        
        self.gail_learner= gail_learner

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        
        self.kl_average = 0
        
    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, pro = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                
                with torch.no_grad():
                    values_g, action_log_probs_g, dist_entropy_g, pro_g = self.gail_learner.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)

                if self.actor_critic.cat == True:
                    x = nn.KLDivLoss()(torch.log(pro),pro_g)
                    cur = x.item()
                else:
                    x = nn.KLDivLoss()(action_log_probs,torch.exp(action_log_probs_g))
                    cur = x.item()
                

                loss = self.log_kl*(self.target_kl - cur)
                self.kl_optim.zero_grad()
                loss.backward()
                self.kl_optim.step()
                self.kl_coef = self.log_kl.exp()/10
                
                
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                    dist_entropy * self.entropy_coef + 
                    self.kl_coef * x).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        print(self.kl_coef)
        self.target_kl += .0005
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
