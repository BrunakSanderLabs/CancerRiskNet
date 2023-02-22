import math
import torch
import torch.nn as nn
import pdb
import numpy as np
from cancerrisknet.models.pools.factory import get_pool
from cancerrisknet.models.abstract_risk_model import AbstractRiskModel
from cancerrisknet.models.factory import RegisterModel


@RegisterModel("mlp")
class MLP(AbstractRiskModel):
    """
        A basic risk model that embeds codes using a multi-layer perception.
    """
    def __init__(self, args):
        super(MLP, self).__init__(args)
        for layer in range(args.num_layers):
            linear_layer = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.add_module('linear_layer_{}'.format(layer), linear_layer)
        self.relu = nn.ReLU()

    def encode_trajectory(self, embed_x, batch=None):
        seq_hidden = embed_x
        for indx in range(self.args.num_layers):
            name = 'linear_layer_{}'.format(indx)
            seq_hidden = self._modules[name](seq_hidden)
            seq_hidden = self.relu(seq_hidden)
        return seq_hidden
