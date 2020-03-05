import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

from torchkge.utils import init_embedding

from torch.utils.data import DataLoader

from torchkge.data.DataLoader import load_fb15k237

def load_embeddings_batch(batch):
    pass 

def compute_adj():
    pass 

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

        self.a_2 = nn.Linear(out_dim, 1)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)


    def forward(self, triplets, ent_embed, rel_embed, adj):

        N = triplets.shape[0]

        h = torch.cat((ent_embed[triplets[0]], rel_embed[triplets[1]], ent_embed[triplets[2]]), dim=1)
        c = self.a(h)
        b = F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)

        temp = triplets.t()
        edges = torch.stack(temp[0], temp[2])

        e_b_sum_ = torch.sparse_coo_tensor(edges, e_b, torch.size(N, N, 1))
        e_b_sum = torch.sparse.sum(e_b_sum_, dim=1)

        temp1 = e_b * c # ?? 
        
        h_ = torch.sparse_coo_tensor(edges, temp1, torch.size(N, N, self.out_dim))
        h_sum = torch.sparse.sum(h_, dim=1)

        out = h_sum.div(e_b_sum)

        if self.concat:
            return F.elu(out)
        else:
            return out 

def loss_func(triplets, neg_sampling_ratio, ent_embed, rel_embed):
    """
    Finalize order of src, dst, rel in triplets
    """
    n = len(triplets)
    if type(triplets) == np.ndarray:
        triplets = torch.from_numpy(triplets)

    pos_triplets = triplets[:n // (neg_sampling_ratio + 1)]
    pos_triplets = torch.cat([pos_triplets for _ in range(neg_sampling_ratio)])

    neg_triplets = triplets[n // (neg_sampling_ratio + 1) : ]

    src_embed = ent_embed[pos_triplets[:, 0]]
    dst_embed = ent_embed[pos_triplets[:, 1]]
    rel_embed = rel_embed[pos_triplets[:, 2]]

    x = src_embed + rel_embed - dst_embed
    pos_norm = torch.norm(x, p=2, dim=1)

    src_embed = ent_embed[neg_triplets[:, 0]]
    dst_embed = ent_embed[neg_triplets[:, 1]]
    rel_embed = rel_embed[neg_triplets[:, 2]]

    x = src_embed + rel_embed - dst_embed
    neg_norm = torch.norm(x, p=2, dim=1)

    y = torch.ones(neg_sampling_ratio * n)

    loss_fn = nn.MarginRankingLoss(margin=5)
    loss = loss_fn(pos_norm, neg_norm, y)

    return loss 