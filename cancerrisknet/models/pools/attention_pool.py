import torch
import torch.nn as nn
from cancerrisknet.models.pools.factory import RegisterPool


@RegisterPool('Softmax_AttentionPool')
class SoftmaxAttentionPool(nn.Module):
    def __init__(self, args):
        super(SoftmaxAttentionPool, self).__init__()
        self.attention_fc = nn.Linear(args.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # X dim: B, C, W, H
        spatially_flat_size = (*x.size()[:2], -1)  # B, C, N
        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1,2))  # B, N, 1
        attention_scores = self.softmax( attention_scores.transpose(1,2))  # B, 1, N
        x = x * attention_scores  # B, C, N
        x = torch.sum(x, dim=-1)
        return x
