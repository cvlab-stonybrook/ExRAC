import math
import torch
import torch.nn as nn
from typing import List

ACTIVATION_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU,
}


class MlpBlock(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim: List[int],
                 out_dim: int,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 last_act = True):
        super().__init__()

        assert activation in ACTIVATION_TYPE.keys()
        self.layers = []
        all_dims = [in_dim]
        all_dims += hidden_dim
        all_dims.append(out_dim)
        for layer_id in range(len(all_dims) - 1):
            dim_in = all_dims[layer_id]
            dim_out = all_dims[layer_id + 1]
            self.layers.append(nn.Linear(dim_in, dim_out))
            if layer_id == len(all_dims) - 2 and not last_act:
                break
            else:
                self.layers.append(ACTIVATION_TYPE[activation]())
        if dropout > 0.0:
            self.layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.layers(x)
        return out
