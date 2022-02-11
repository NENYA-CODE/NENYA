# NENYA

This is the source code of NENYA. 

The proposed NENYA is an end-to-end mitigation solution for a large-scale database system powered by a novel cascade reinforcement learning model. By taking the states of databases as input, NENYA directly outputs mitigation actions and is optimized based on jointly cumulative feedback on mitigation costs and failure rates. As the overwhelming majority of databases do not require mitigation actions, NENYA utilizes a novel cascade decision structure to firstly reliably filter out such databases and then focus on choosing appropriate mitigation actions for the rest.


# File structure



# Run 


## Requirements

- python >= 3.6.0
- pytorch >= 1.6.0


## Run experiments
~~~
python main.py
~~~

- if you want to change any options, check "python main.py --help"
  - you train and test agent using main.py

- '--env_name', type=str, default = "env_Mitigation"
- '--train', type=bool, default=True, help="(default: True)"
- '--render', type=bool, default=False, help="(default: False)"
- '--epochs', type=int, default=1000, help='number of epochs, (default: 1000)'
- '--entropy_coef', type=float, default=1e-2, help='entropy coef (default : 0.01)'
- '--critic_coef', type=float, default=0.5, help='critic coef (default : 0.5)'
- '--learning_rate', type=float, default=3e-4, help='learning rate (default : 0.0003)'
- '--gamma', type=float, default=0.99, help='gamma (default : 0.99)'
- '--lmbda', type=float, default=0.95, help='lambda using GAE(default : 0.95)'
- '--eps_clip', type=float, default=0.2, help='actor and critic clip range (default : 0.2)'
- '--K_epoch', type=int, default=64, help='train epoch number(default : 10)'
- '--T_horizon', type=int, default=2048, help='one generation before training(default : 72)'
- '--hidden_dim', type=int, default=64, help='actor and critic network hidden dimension(default : 64)'
- '--minibatch_size', type=int, default=64, help='minibatch size(default : 64)'
- '--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)'
- '--load', type=str, default = 'no', help = 'load network name in ./model_weights'
- '--save_interval', type=int, default = 100, help = 'save interval(default: 100)'
- '--print_interval', type=int, default = 20, help = 'print interval(default : 20)'
- '--use_cuda', type=bool, default = True, help = 'cuda usage(default : True)'