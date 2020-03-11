import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

from typing import Dict, List

from utils import create_mappings, get_batch_neighbors, rel2edge


class KGATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_rels, dropout=0.5, first=False, device='cpu'):
        super().__init__()

        self.first = first
        self.dropout = dropout

        self.ent_embed = nn.Parameter(torch.randn(num_nodes, input_dim)).to(device)
        self.rel_embed = nn.Parameter(torch.randn((num_rels, input_dim))).to(device)

        self.fc_ent = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc_rel = nn.Linear(input_dim, hidden_dim).to(device)

        self.fc_rel2 = nn.Linear(hidden_dim, output_dim).to(device)
        self.fc_rel3 = nn.Linear(3 * output_dim, output_dim).to(device)

        self.a = nn.Linear(output_dim, 1).to(device)
        self.fc = nn.Linear(3 * hidden_dim, output_dim).to(device)

    def init_weights(self, ent_embed, rel_embed):
        if type(ent_embed) != torch.Tensor or type(rel_embed) != torch.Tensor:
            ent_embed = torch.from_numpy(ent_embed)
            rel_embed = torch.from_numpy(rel_embed)

        self.ent_embed.data = ent_embed
        self.rel_embed.data = rel_embed

    def save_weights(self):
        pass

    def forward(self, triplets: torch.Tensor, ent_emb=None, rel_emb=None):
        """
        triplets: shape[batch_size, 3]
        """
        src_, dst_, rel_ = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        src_ = [s.item() for s in src_]
        dst_ = [d.item() for d in dst_]
        rel_ = [r.item() for r in rel_]

        nodes = list(set(src_ + dst_))

        #precompute
        mapping = create_mappings(src_, dst_)
        neighbors = get_batch_neighbors(src_, dst_)

        #precompute
        neighbors = {mapping[key]: [mapping[val] for val in neighbors[key]] for key, val in zip(neighbors.keys(), neighbors.values())}
        # Add multi-hop neighbors
        # Add neighbor sampling

        if self.first:
            src, dst, rel = self.ent_embed[src_], self.ent_embed[dst_], self.rel_embed[rel_]
            # print(src)
        else:
            assert ent_emb != None
            assert rel_emb != None
            # src, dst, rel = ent_emb[src_], ent_emb[dst_], rel_emb[rel_]
            src = ent_emb[[mapping[s.item()] for s in src_]]
            dst = ent_emb[[mapping[d.item()] for d in dst_]]

            rels = {j: i for i, j in enumerate(set([r.item() for r in rel_]))}
            rel = rel_emb[[rels[r.item()] for r in rel_]]

        src = F.dropout( self.fc_ent(src), self.dropout )
        dst = F.dropout( self.fc_ent(dst), self.dropout )
        rel = F.dropout( self.fc_rel(rel), self.dropout )


        c = self.fc(torch.cat([src, dst, rel], dim=1))

        b = F.leaky_relu(self.a(c)).squeeze()
        b = torch.exp(b)
        # b_sum = torch.stack([sum([b[n_] for n_ in neighbors[n]]) for n in [mapping[s.item()] for s in src_]]) # phew
        b_sum = torch.stack([ torch.sum( b[ [n for n in neighbors[mapping[s]]] ] )  for s in src_])

        alpha = b / b_sum

        # nodes = list(set([s.item() for s in src_] + [d.item() for d in dst_]))
        nodes = [mapping[node] for node in nodes]
        ac = alpha.unsqueeze(1) * c
        # h_ent = torch.stack([sum([alpha[n_] * c[n_] for n_ in neighbors[n]]) for n in nodes])
        # h_ent = [ [ac [n_ for n_ in neighbors[n]]] for n in nodes]
        h_ent = torch.stack( [   torch.sum (ac[ [n_ for n_ in neighbors[n]] ], dim=0 ) for n in nodes] )


        # Relations embeddings updated from new node embeddings
        rel = self.fc_rel2(rel)
        # r2e = rel2edge(src_, dst_, rel_)

        # rel = torch.stack ( [
        #     torch.mean( torch.stack([ torch.cat([
        #         h_ent[mapping[n[0]]], h_ent[mapping[n[1]]], rel[r] ])  for n in r2e[r]]  ), dim=0 ) for r in r2e.keys()] )

        # h_rel = self.fc_rel3(rel)
        # return h_ent, h_rel
        # print("done")

        return h_ent, rel


class KGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_nodes, num_rels, device='cpu'):
        super(KGAT, self).__init__()

        self.a = nn.ModuleList([
            KGATLayer(input_dim, hidden_dim, output_dim, num_nodes, num_rels, first=True, device=device).to(device)
            for _ in range(num_heads)
        ])

        self.a2 = KGATLayer(num_heads * output_dim, output_dim, output_dim, num_nodes, num_rels, device=device).to(device)

    def forward(self, triplets: torch.Tensor, ent_embed: torch.Tensor, rel_embed: torch.Tensor):
        h_ents, h_rels = [], []
        for layer in self.a:
            layer.init_weights(ent_embed, rel_embed)
            h_ent, h_rel = layer(triplets)
            h_ents.append(h_ent)
            h_rels.append(h_rel)
            # print(h_ent.shape)
            # print(h_rel.shape)

        h_ents = torch.mean(h_ents, dim=1)
        h_rels = torch.mean(h_rels, dim=1)

        # h_ent, h_rel = self.a2(triplets, h_ents, h_rels)

        return h_ents, h_rels

