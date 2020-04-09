import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

import numpy as np

class SpecialSpmmFunctionFinal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(torch.cuda.is_available()):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None

class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


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

        self.special_spmm_final = SpecialSpmmFinal()


        self.ent_embed = torch.randn(n_entities, out_dim).to(device).detach()
        self.rel_embed = torch.randn(n_relations, out_dim).to(device).detach()

        # self.rel_out = nn.Linear(out_dim, out_dim)
        # nn.init.xavier_normal_(self.rel_out.weight, gain=1.414)
        # self.ent_out = nn.Linear(out_dim, out_dim)
        # nn.init.xavier_normal_(self.ent_out.weight, gain=1.414)

    def get_embeddings(self):
        return self.ent_embed, self.rel_embed
        # return self.ent_out.weight, self.rel_out.weight

    def forward(self, triplets, ent_embed, rel_embed, nodes_=None, edges_=None):
        triplets = triplets.to(self.device)
        ent_embed = ent_embed.to(self.device)
        rel_embed = rel_embed.to(self.device)

        # nodes = list(set([t.item() for t in torch.cat(( triplets[:, 0], triplets[:, 1] ))]))
        # edges = list(set([t.item() for t in triplets[:, 2]]))

        N = self.n_entities

        h = torch.cat((ent_embed[triplets[:, 0]], rel_embed[triplets[:, 2]], ent_embed[triplets[:, 1]]), dim=1)
        c = self.a(h)
        b = F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[1]])

        # e_b_sum_ = torch.sparse_coo_tensor(edges, e_b, torch.Size((N, N, 1)))
        # e_b_sum_ = e_b_sum_.detach()
        # e_b_sum = torch.sparse.sum(e_b_sum_, dim=1)

        e_b_sum = self.special_spmm_final(edges, e_b, N, e_b.shape[0], 1)

        temp1 = e_b * c

        # h_ = torch.sparse_coo_tensor(edges, temp1, torch.Size((N, N, self.out_dim)))
        # h_ = h_.detach()
        # h_sum = torch.sparse.sum(h_, dim=1)

        h_sum = self.special_spmm_final(edges, temp1,  N, e_b.shape[0], self.out_dim)

        hs = h_sum
        ebs = e_b_sum
        ebs[ebs == 0] = 1e-12

        # out = h_sum.div(e_b_sum)
        h_ent = hs / ebs

        index = triplets[:, 2]
        h_rel = scatter(temp1, index=index, dim=0, reduce="mean") # SUCH A LIFESAVER!

        # h_ent = h_ent_

        # Hacky AF. Sigh
        self.ent_embed[nodes_] = h_ent[nodes_]
        # h_ent = self.ent_embed

        self.rel_embed[edges_] = h_rel[edges_]
        # h_rel = self.rel_embed

        # h_ent = self.ent_out(h_ent)
        # h_rel = self.rel_out(h_rel)

        if self.concat:
            return F.elu(h_ent), F.elu(h_rel)
        else:
            return h_ent, h_rel

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

    def get_embeddings(self):
        return self.att2.get_embeddings()

    def forward(self, triplets, ent_embed, rel_embed, nodes=None, edges=None):
        att1_out = [a(triplets, ent_embed, rel_embed, nodes, edges) for a in self.att1]
        h_ent = torch.cat([a[0] for a in att1_out], dim=1)
        h_rel = torch.cat([a[1] for a in att1_out], dim=1)
        # rel_embed = self.fc1(rel_embed)
        # out, rel_embed = self.att2(triplets, att1_out, rel_embed)
        # rel_embed = self.fc2(rel_embed)

        h_ent, h_rel = self.att2(triplets, h_ent, h_rel, nodes, edges)

        # return F.softmax(h_ent, dim=-1), F.softmax(h_rel, dim=-1)
        return h_ent, h_rel
