import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class KGAtt(nn.Module):
    """
    Single Hop Version
    """
    def __init__(self, n_entities, n_relations, in_dim, out_dim, concat=True, device='cpu'):
        super(KGAtt, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.a = nn.Linear(3 * in_dim, out_dim).to(device)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.concat = concat

        self.a_2 = nn.Linear(out_dim, 1).to(device)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

        # self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, triplets, ent_embed, rel_embed):
        triplets = triplets.to(self.device)
        ent_embed = ent_embed.to(self.device)
        rel_embed = rel_embed.to(self.device)

        N = self.n_entities

        h = torch.cat((ent_embed[triplets[:, 0]], rel_embed[triplets[:, 2]], ent_embed[triplets[:, 1]]), dim=1).to(self.device)
        c = self.a(h)
        b = F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[2]])

        e_b_sum_ = torch.sparse_coo_tensor(edges, e_b, torch.Size((N, N, 1)))
        e_b_sum_ = e_b_sum_.detach()
        e_b_sum = torch.sparse.sum(e_b_sum_, dim=1)

        # e_b_sum = self.special_spmm_final(edges, e_b, N, e_b.shape[0], 1)

        temp1 = e_b * c

        h_ = torch.sparse_coo_tensor(edges, temp1, torch.Size((N, N, self.out_dim)))
        h_ = h_.detach()
        h_sum = torch.sparse.sum(h_, dim=1)

        hs = h_sum.to_dense()
        ebs = e_b_sum.to_dense()
        ebs[ebs == 0] = 1e-12

        # out = h_sum.div(e_b_sum)
        out = hs / ebs

        if self.concat:
            return F.elu(out)
        else:
            return out, rel_embed


class MultiHeadKGAtt(nn.Module):
    def __init__(self, n_entities, n_relations, in_dim, hidden_dim, out_dim, num_heads, device='cpu'):
        super(MultiHeadKGAtt, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.device = device

        self.att1 = nn.ModuleList([KGAtt(n_entities, n_relations, in_dim, hidden_dim, device=device).to(device) for n in range(num_heads)])
        self.att2 = KGAtt(n_entities, n_relations, num_heads * hidden_dim, out_dim, concat=False, device=device).to(device)

        self.fc1 = nn.Linear(in_dim, num_heads * hidden_dim).to(device)
        self.fc2 = nn.Linear(num_heads * hidden_dim, out_dim).to(device)

    def forward(self, triplets, ent_embed, rel_embed):
        att1_out = torch.cat([a(triplets, ent_embed, rel_embed) for a in self.att1], dim=1)
        rel_embed = self.fc1(rel_embed)
        out, rel_embed = self.att2(triplets, att1_out, rel_embed)
        rel_embed = self.fc2(rel_embed)

        return out, rel_embed
