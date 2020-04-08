"""
Modified EmbeddingMul layer
Adapted from code by Noémien Kocher
"""


import torch
import torch.nn as nn 
from functools import reduce
from operator import mul 

class EmbeddingMul2(nn.Module):
    def __init__(self, num_ents, num_dims, embed, requires_grad, device):
        super(EmbeddingMul2, self).__init__() 

        self.num_ents = num_ents
        self.num_dims = num_dims
        self.device = device

        self.requires_grad = requires_grad

        # self.ones = torch.eye(num_ents, requires_grad=False, device=device)
        self.embedding = nn.Parameter(data = torch.randn(num_ents, num_dims)).to(device)
    
    def forward(self, input_):
        last_oh = self.to_one_hot(input_)

        with torch.set_grad_enabled(self.requires_grad):
            result = torch.stack(
                [torch.mm(batch.float(), self.embedding)
                 for batch in last_oh], dim=0)
        return result.squeeze()
    
    def to_one_hot(self, input_):
        ones = torch.eye(self.num_ents, requires_grad=False, device=self.device)
        # indices = torch.tensor([[i, i] for i in range(self.num_ents)]).T
        # values = torch.tensor([1 for i in range(self.num_ents)])
        # ones = torch.sparse_coo_tensor(indices, values, size=(self.num_ents, self.num_ents)).to("cuda")
        result = torch.index_select(
            ones, 0, input_.view(-1).long())
        del ones 
        result = result.view(input_.size() + (self.num_ents,))
        result = result.to(torch.float)
        result.requires_grad = True

        return result