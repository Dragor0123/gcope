import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot


class GPF(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p

