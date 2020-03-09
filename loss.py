import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_func(triplets, neg_sampling_ratio, ent_embed, rel_embed, device='cpu'):
    """
    Triplets order: src, dst, rel
    """
    n = len(triplets)
    if type(triplets) == np.ndarray:
        triplets = torch.from_numpy(triplets)

    pos_triplets = triplets[:n // (neg_sampling_ratio + 1)]
    pos_triplets = torch.cat([pos_triplets for _ in range(neg_sampling_ratio)])

    neg_triplets = triplets[n // (neg_sampling_ratio + 1):]

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

    # y = torch.ones(neg_sampling_ratio * n)
    y = torch.ones(len(pos_triplets)).to(device)

    loss_fn = nn.MarginRankingLoss(margin=5)
    loss = loss_fn(pos_norm, neg_norm, y)

    return loss
