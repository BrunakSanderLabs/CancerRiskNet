import math
import torch
import torch.nn as nn
import pdb
import numpy as np
from cancerrisknet.models.pools.factory import get_pool
from cancerrisknet.models.abstract_risk_model import AbstractRiskModel
from cancerrisknet.models.factory import RegisterModel


class AbstractRNN(AbstractRiskModel):
    """
        The abstract risk model which embeds codes using a recurrent neural network (GRU or LSTM).
    """

    def __init__(self, args):

        super(AbstractRNN, self).__init__(args)

        # Always use bidir RNNs
        assert args.hidden_dim % 2 == 0
        self.hidden_dim = args.hidden_dim // 2

    def encode_trajectory(self, embed_x, batch=None):
        seq_hidden, _ = self.rnn(embed_x)
        return seq_hidden


@RegisterModel("gru")
class GRU(AbstractRNN):
    def __init__(self, args):
        super(GRU, self).__init__(args)
        self.rnn = nn.GRU(input_size=args.hidden_dim, hidden_size=self.hidden_dim, num_layers=args.num_layers,
                          bidirectional=True, batch_first=True, dropout=args.dropout)


@RegisterModel("lstm")
class LSTM(AbstractRNN):
    def __init__(self, args):
        super(LSTM, self).__init__(args)
        self.rnn = nn.LSTM(input_size=args.hidden_dim, hidden_size=self.hidden_dim, num_layers=args.num_layers,
                           bidirectional=True, batch_first=True, dropout=args.dropout)
