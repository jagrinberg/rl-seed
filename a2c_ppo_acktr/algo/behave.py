import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def behavioral_cloning(actor_critic, expert_loader, device, obsfilt=None):
    for expert_batch in expert_loader:
        expert_state, expert_action = expert_batch
        expert_state = obsfilt(expert_state.numpy(), update=False)
        expert_state = torch.FloatTensor(expert_state).to(device)
        actor_probs = actor_critic.get_prob(expert_state)
        expert_action = expert_action.to(device)

        loss = nn.BCELoss()(actor_probs, expert_action)
        return loss