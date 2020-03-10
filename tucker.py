import numpy as np
import torch
import torch.nn as nn


class TuckER(nn.Module):
    def __init__(self, num_entities, num_relations, inp_dim, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5, device='cpu'):

        super(TuckER, self).__init__()

        self.E = nn.Embedding(num_entities, inp_dim, padding_idx=0)
        self.R = nn.Embedding(num_relations, inp_dim, padding_idx=0)
        self.W = nn.Parameter(torch.randn((inp_dim, inp_dim, inp_dim), dtype=torch.float, device=device, requires_grad=True))

        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)

        self.loss = nn.BCELoss()

        self.bn0 = nn.BatchNorm1d(inp_dim)
        self.bn1 = nn.BatchNorm1d(inp_dim)
    
    # def forward(self, )
