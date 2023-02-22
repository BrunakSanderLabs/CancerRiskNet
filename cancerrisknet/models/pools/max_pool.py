import torch
import torch.nn as nn
from cancerrisknet.models.pools.factory import RegisterPool


@RegisterPool('GlobalMaxPool')
class GlobalMaxPool(nn.Module):
    def __init__(self, args):
        super(GlobalMaxPool, self).__init__()

    def forward(self, x):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x, _ = torch.max(x, dim=-1)
        return x
