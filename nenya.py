from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class NENYA(nn.Module):
    def __init__(self, device, state_dim, action_dim, gamma,q_lr,actor_lr,soft_update_rate,hidden_dim,learn_start_size,memory_size,batch_size,
    layer_num,activation_function,last_activation,trainable_std,on_policy, noise):
        super(NENYA,self).__init__()
        self.device = device

        self.gamma = gamma
        self.q_lr = q_lr
        self.actor_lr = actor_lr
        self.soft_update_rate = soft_update_rate
        self.hidden_dim = hidden_dim
        self.learn_start_size = learn_start_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.activation_function = activation_function
        self.last_activation = last_activation
        self.trainable_std = trainable_std
        self.on_policy = on_policy
    
        self.actor = Actor(self.layer_num, state_dim, action_dim, self.hidden_dim, \
                           self.activation_function, self.last_activation, self.trainable_std)
        
        self.target_actor = Actor(self.layer_num, state_dim, action_dim, self.hidden_dim, \
                           self.activation_function, self.last_activation, self.trainable_std)
        
        self.q = Critic(self.layer_num, state_dim+action_dim, 1, self.hidden_dim, self.activation_function, None)
        
        self.target_q = Critic(self.layer_num, state_dim+action_dim, 1, self.hidden_dim, self.activation_function, None)
        
        self.soft_update(self.q, self.target_q, 1.)
        self.soft_update(self.actor, self.target_actor, 1.)
        
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.q_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.memory_size), state_dim = state_dim, num_action = action_dim)
        
        self.noise = noise
        
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)
            
    def get_action(self,x):
        action = self.actor(x)

        return action

 
    
    def put_data(self,transition):
        self.data.put_data(transition)
        
    def train_net(self, batch_size, n_epi):
        data = self.data.sample(shuffle = True, batch_size = batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])

        targets = rewards + self.gamma * (1 - dones) * self.target_q(next_states, self.target_actor(next_states))
        q_loss = F.smooth_l1_loss(self.q(states,actions), targets.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        actor_loss = - self.q(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.q, self.target_q, self.soft_update_rate)
        self.soft_update(self.actor, self.target_actor, self.soft_update_rate)

    @torch.no_grad()
    def get_state_emb(self,x):
        return self.actor(x)