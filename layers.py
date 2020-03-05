import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

class KGAtt(nn.Module):
    """
    Single Hop Version
    """
    def __init__(self, n_entities, n_relations, in_dim, out_dim, num_heads, concat=True):
        super(KGAtt, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.a = nn.Linear(3*in_dim, out_dim)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.concat = concat

        self.a_2 = nn.Linear(out_dim, 1)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)


    def forward(self, triplets, ent_embed, rel_embed):

        # N = triplets.shape[0]
        N = self.n_entities
        print(triplets.shape)

        h = torch.cat((ent_embed[triplets[:, 0]], rel_embed[triplets[:, 2]], ent_embed[triplets[:, 1]]), dim=1)
        c = self.a(h)
        b = F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)
        # e_b = e_b.squeeze()

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[2]])

        e_b_sum_ = torch.sparse_coo_tensor(edges, e_b, torch.Size((N, N, 1)))
        e_b_sum = torch.sparse.sum(e_b_sum_)

        temp1 = e_b * c # ?? 
        
        h_ = torch.sparse_coo_tensor(edges, temp1, torch.Size((N, N, self.out_dim)))
        h_sum = torch.sparse.sum(h_, dim=1)

        out = h_sum.div(e_b_sum)

        if self.concat:
            return F.elu(out)
        else:
            return out 

