import math
import torch
import torch.nn as nn
import pdb
import numpy as np
from cancerrisknet.models.pools.factory import get_pool
from cancerrisknet.models.factory import RegisterModel
from cancerrisknet.models.utils import OneHotLayer
import torch.nn.functional as F
from cancerrisknet.models.abstract_risk_model import AbstractRiskModel


class OneHotRiskModel(AbstractRiskModel):
    """
        The abstract model class for regression models that use one hot embedding.
    """

    def __init__(self, args):
        
        super(OneHotRiskModel, self).__init__(args)
        self.args = args
        self.code_embed = OneHotLayer(num_classes=self.vocab_size, padding_idx=0)
        num_features_in = self.vocab_size + 1 if args.add_age_neuron else self.vocab_size
        self.add_module('linear_layer', nn.Linear(num_features_in, 1, bias=False))
        self.pool = get_pool(args.pool_name)(args)

    def forward(self, x, batch=None):
        # Overrides forward() and skip char and time embedding 
        embed_x = self.code_embed(x)
        seq_hidden = self.pool(embed_x.transpose(1, 2))
        if self.args.add_age_neuron:
            age_in_year = batch['age']/365.
            seq_hidden = torch.cat((seq_hidden, age_in_year), axis=1)
        seq_hidden = self.dropout(seq_hidden)
        seq_hidden = self._modules['linear_layer'](seq_hidden)
        logit = self.get_cox_logit(seq_hidden)

        return logit


@RegisterModel("bow")
class ProportionalHazards(OneHotRiskModel):
    """
        The bag-of-words model which encodes the events by direct pooling and use a linear layer to compute risk.
    """
    def __init__(self, args):

        assert args.pool_name != 'Softmax_AttentionPool', "COX models are not compatible with attention poolings"
        super(ProportionalHazards, self).__init__(args)
        self.add_module('linear_layer_hz', nn.Linear(1, len(self.args.month_endpoints), bias=False))
        self.exp = torch.exp
    
    def get_cox_logit(self, seq_hidden):
        hazard_ratio = self.exp(seq_hidden)
        hazards = self._modules['linear_layer_hz'](hazard_ratio)
        return hazards


@RegisterModel("addition_bow")
class AdditiveHazards(OneHotRiskModel):
    """
        A simplified bag-of-words model which encodes the events by direct pooling and use a non-parameterized
        additional layer to compute risk.
    """

    def __init__(self, args):

        assert args.pool_name != 'Softmax_AttentionPool', "COX models are not compatible with attention poolings"
        super(AdditiveHazards, self).__init__(args)
        self.baseline_hazard = nn.Parameter(torch.zeros(len(self.args.month_endpoints)))
        self.register_parameter('baseline_hazard', self.baseline_hazard)

    def get_cox_logit(self, seq_hidden):
        hazards = self.baseline_hazard + seq_hidden
        return hazards
