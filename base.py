from abc import *
import numpy as np
import torch
import torch.nn as nn


class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()
    @abstractmethod
    def forward(self, x):
        return x
    
class Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.relu,last_activation = None):
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l2_1 = nn.Linear(hidden_dim,2)
        self.l2_2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3_1 = nn.Linear(hidden_dim,2)
        self.l3_2 = nn.Linear(hidden_dim,hidden_dim)
        self.l4 = nn.Linear(hidden_dim,output_dim)
        self.network_init()
        self.relu = nn.ReLU()
        self.sigmiod = nn.Sigmoid()
        self.tahn = nn.Tanh()
        self.softmax = nn.Softmax()


        
    def forward(self, x):
        return self._forward(x)
    def _forward(self, x):
        action_list = []
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        a1 = self.sigmiod(self.l2_1(x))
        x = self.tahn(self.l2_2(x))
        a2 = self.sigmiod(self.l3_1(x))
        x = self.tahn(self.l3_2(x))
        a3 = self.softmax(self.l4(x))

        for i in range(x.shape[0]):
            if a1[i] > 0.5:
                action_list.append(3)
            elif a2[i] > 0.5:
                action_list.append(3)
            else:
                action_list.append(torch.argmax(a3[i]))

        return torch.tensor(np.array(action_list))
   
    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 