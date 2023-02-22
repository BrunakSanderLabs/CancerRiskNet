import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from cancerrisknet.models.pools.factory import get_pool
from cancerrisknet.models.factory import RegisterModel
from cancerrisknet.models.abstract_risk_model import AbstractRiskModel


@RegisterModel("transformer")
class Transformer(AbstractRiskModel):
    def __init__(self, args):

        super(Transformer, self).__init__(args)

        for layer in range(args.num_layers):
            transformer_layer = TransformerLayer(args)
            self.add_module('transformer_layer_{}'.format(layer), transformer_layer)

    def encode_trajectory(self, embed_x, batch=None):
        """
            Computes a forward pass of the model.

            Returns:
                The result of feeding the input through the model.
        """

        # Run through transformer
        seq_x = embed_x
        for indx in range(self.args.num_layers):
            name = 'transformer_layer_{}'.format(indx)
            seq_x = self._modules[name](seq_x)
        return seq_x


class TransformerLayer(nn.Module):
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        self.args = args
        self.multihead_attention = MultiHeadAttention(self.args)
        self.layernorm_attn = nn.LayerNorm(self.args.hidden_dim)
        self.fc1 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.layernorm_fc = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, x):
        h = self.multihead_attention(x)
        x = self.layernorm_attn(h + x)
        h = self.fc2(self.relu(self.fc1(x)))
        x = self.layernorm_fc(h + x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        assert args.hidden_dim % args.num_heads == 0

        self.query = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout = nn.Dropout(p=args.dropout)

        self.dim_per_head = args.hidden_dim // args.num_heads

        self.aggregate_fc = nn.Linear(args.hidden_dim, args.hidden_dim)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, x):
        B, N, H = x.size()

        # perform linear operation and split into h heads
        k = self.key(x).view(B, N, self.args.num_heads, self.dim_per_head)
        q = self.query(x).view(B, N, self.args.num_heads, self.dim_per_head)
        v = self.value(x).view(B, N, self.args.num_heads, self.dim_per_head)

        # transpose to get dimensions B * args.num_heads * S * dim_per_head
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        h = self.attention(q, k, v)

        # concatenate heads and put through final linear layer
        h = h.transpose(1, 2).contiguous().view(B, -1, H)

        output = self.aggregate_fc(h)
        return output
