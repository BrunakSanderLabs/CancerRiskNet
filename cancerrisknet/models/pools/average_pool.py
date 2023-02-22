import torch
import torch.nn as nn
from cancerrisknet.models.pools.factory import RegisterPool


@RegisterPool('GlobalAvgPool')
class GlobalAvgPool(nn.Module):

    def __init__(self, args):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x = torch.mean(x, dim=-1)
        return x
