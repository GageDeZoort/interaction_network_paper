import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.output_size = output_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),            
        )

    def forward(self, m):
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, C):
        return self.layers(C)

class InteractionNetwork(nn.Module):
    def __init__(self, object_dim, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()

        self.phi_R1 = RelationalModel(2*object_dim + relation_dim, effect_dim, 8)
        self.phi_R2 = RelationalModel(2*object_dim + relation_dim, 1, 8)
        self.phi_O = ObjectModel(object_dim + effect_dim, 8)

    def forward(self, X, Ra, Ro, Ri):
        X = torch.transpose(X, 1, 2)

        # first marshalling step: build interaction terms
        Xi = torch.bmm(Ro, X)
        Xo = torch.bmm(Ri, X)
        m1 = torch.cat([Xi, Xo, Ra], dim=2)

        # relational model: determine effects
        E = self.phi_R1(m1)

        # aggregation step: aggregate effects
        A = torch.bmm(torch.transpose(Ri, 1, 2), E)
        C = torch.cat([X, A], dim=2)

        # object model: re-embed hit features
        X_tilde = self.phi_O(C)

        # re-marshalling step: build new interaction terms
        Xi_tilde = torch.bmm(Ri, X_tilde)
        Xo_tilde = torch.bmm(Ro, X_tilde)
        m2 = torch.cat([Xi_tilde, Xo_tilde, E], dim=2)

        W = torch.sigmoid(self.phi_R2(m2))
        #W = torch.sigmoid(self.phi_R1(m2))
        return W
