from configparser import ConfigParser
from argparse import ArgumentParser
from env_mitigation import MitigationEnv
import torch
import numpy as np
import os


from nenya import NENYA

import random
import pandas as pd

from utils import make_transition, Dict, RunningMeanStd
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--algo", type=str, default = 'nenya', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=2000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
    

    
env = MitigationEnv()
action_dim = env.action_space
state_dim = env.state_space
state_rms = RunningMeanStd(state_dim)


gamma = 0.99
q_lr = 3e-4
actor_lr = 3e-4
soft_update_rate = 0.005
hidden_dim = 256
learn_start_size = 1000
memory_size = 1e+6
batch_size = 64
layer_num = 3
activation_function = torch.relu
last_activation = torch.tanh
trainable_std = False
on_policy = False


if args.algo == 'ppo' :
    agent = PPO(device, state_dim, action_dim, gamma,q_lr,actor_lr,soft_update_rate,hidden_dim,learn_start_size,memory_size,batch_size,
    layer_num,activation_function,last_activation,trainable_std,on_policy)
elif args.algo == 'sac' :
    agent = SAC(device, state_dim, action_dim, agent_args)
elif args.algo == 'NENYA' :
    from utils.noise import OUNoise
    noise = OUNoise(action_dim,0)
    agent = NENYA(device, state_dim, action_dim, gamma,q_lr,actor_lr,soft_update_rate,hidden_dim,learn_start_size,memory_size,batch_size,
    layer_num,activation_function,last_activation,trainable_std,on_policy, noise=noise)

    
if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/"+args.load))
    
score_lst = []
state_lst = []

c_all_r = []
f_all_r = []
for n_epi in range(args.epochs):
    score = 0.0
    env.reset()
    
    state = env.initial_state
    done = False
    while not done:
        actions = []
        state_indexes = np.where(state[:,-1]!=-1)
        active_state_indexes = np.where(state[:,-1]==1)
        assigned_states = []
        count_action2 = 0

        for j in list(state_indexes[0]):
            action, _ = agent.get_action(torch.from_numpy(state[j,:32]).float().to(device))
            act = action.cpu().detach().numpy()
            if j not in assigned_states:
                if act == 0:
                    if state[j,-1] == 0.0:
                        actions.append([j,3]) 
                        assigned_states.append(j)
                    else:
                        assigned_states.append(j) 
                        state_cadidate = list(np.where(state[:, -1] == 0.0)[0])
                        state_cadidate = set(state_cadidate).difference(set(assigned_states))
                        if len(state_cadidate)>0:
                            disk_ind = random.sample(state_cadidate,1)[0]
                            actions.append([j,act]) 
                            actions.append([disk_ind, 1])
                            assigned_states.append(disk_ind)
                        else: 
                            actions.append([j,3]) 

                elif act == 1:
                    if state[j, -1] == 1.0:
                        actions.append([j, 3])
                        assigned_states.append(j)
                    else:
                        assigned_states.append(j)
                        state_cadidate = list(np.where(state[:, -1] == 1.0)[0])
                        state_cadidate = set(state_cadidate).difference(set(assigned_states))
                        if len(state_cadidate)>0:
                            actions.append([j, act])
                            disk_ind = random.sample(state_cadidate,1)[0]
                        
                            actions.append([disk_ind, 0])
                            assigned_states.append(disk_ind)
                        else: 
                            actions.append([j,3]) 
                else:
                    actions.append([j, act])
                    assigned_states.append(j)
        
        next_states_all, all_costs, done = env.step(actions)
        state = env.all_state[:, env.time_step,:]
        count = 0
        for index, a in actions:
            st = env.all_state[index, env.time_step-1,:32]
            action = a
            reward = all_costs[count]
            next_st = env.all_state[index, env.time_step,:32]
            if env.all_state[index, env.time_step,-1]==-1:
                done_1 = True
            else:
                done_1 = done
            count += 1
            
            transition = make_transition(st,\
                                        action,\
                                        np.array([reward]),\
                                        next_st,\
                                        np.array([done_1])\
                                        )
            agent.put_data(transition)
            score += reward
        if agent.data.data_idx > learn_start_size: 
            agent.train_net(batch_size, n_epi)
    score_lst.append(score)
    c_all_r.append(env.total_capacity)
    f_all_r.append(env.totoal_faliure)

    env.all_state[:,:,-1][env.all_state[:,:,-1]==-1] = 0
    active_failure_rate = np.multiply(env.all_state[:,:,-1], env.all_state[:,:,-2])



