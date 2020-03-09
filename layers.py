import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class KGAtt(nn.Module):
    """
    Single Hop Version
    """
    def __init__(self, n_entities, n_relations, in_dim, out_dim, concat=True):
        super(KGAtt, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.a = nn.Linear(3 * in_dim, out_dim)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.concat = concat

        self.a_2 = nn.Linear(out_dim, 1)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

    def forward(self, triplets, ent_embed, rel_embed):

        # N = triplets.shape[0]
        N = self.n_entities
        # print(triplets.shape)

        h = torch.cat((ent_embed[triplets[:, 0]], rel_embed[triplets[:, 2]], ent_embed[triplets[:, 1]]), dim=1)
        c = self.a(h)
        b = F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)
        # e_b = e_b.squeeze()

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[2]])

        e_b_sum_ = torch.sparse_coo_tensor(edges, e_b, torch.Size((N, N, 1)))
        e_b_sum_ = e_b_sum_.detach()
        e_b_sum = torch.sparse.sum(e_b_sum_)
        # print(type(e_b_sum))
        # print(e_b_sum.__dict__)

        temp1 = e_b * c  # ??

        h_ = torch.sparse_coo_tensor(edges, temp1, torch.Size((N, N, self.out_dim)))
        h_ = h_.detach()
        h_sum = torch.sparse.sum(h_, dim=1)

        out = h_sum.div(e_b_sum)

        if self.concat:
            return F.elu(out.to_dense())
        else:
            return out.to_dense(), rel_embed


class MultiHeadKGAtt(nn.Module):
    def __init__(self, n_entities, n_relations, in_dim, hidden_dim, out_dim, num_heads):
        super(MultiHeadKGAtt, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.att1 = nn.ModuleList([KGAtt(n_entities, n_relations, in_dim, hidden_dim) for n in range(num_heads)])
        self.att2 = KGAtt(n_entities, n_relations, num_heads * hidden_dim, out_dim, concat=False)

        self.fc1 = nn.Linear(in_dim, num_heads * hidden_dim)
        self.fc2 = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, triplets, ent_embed, rel_embed):
        att1_out = torch.cat([a(triplets, ent_embed, rel_embed) for a in self.att1], dim=1)
        rel_embed = self.fc1(rel_embed)
        out, rel_embed = self.att2(triplets, att1_out, rel_embed)
        rel_embed = self.fc2(rel_embed)
        # print(out.shape)
        return out, rel_embed
