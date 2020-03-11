import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import create_mappings

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

    mapping = create_mappings(pos_triplets[:, 0], pos_triplets[:, 1])
    src_embed = ent_embed[[mapping[i.item()] for i in pos_triplets[:, 0]]]
    dst_embed = ent_embed[[mapping[i.item()] for i in pos_triplets[:, 1]]]

    rels = {j: i for i, j in enumerate(set([r.item() for r in pos_triplets[:, 2]]))}
    rel_embed = rel_embed[[rels[r.item()] for r in pos_triplets[:, 2]]]

    x = src_embed + rel_embed - dst_embed
    pos_norm = torch.norm(x, p=2, dim=1)

    mapping = create_mappings(neg_triplets[:, 0], neg_triplets[:, 1])
    src_embed = ent_embed[[mapping[i.item()] for i in neg_triplets[:, 0]]]
    dst_embed = ent_embed[[mapping[i.item()] for i in neg_triplets[:, 1]]]

    rels = {j: i for i, j in enumerate(set([r.item() for r in neg_triplets[:, 2]]))}
    rel_embed = rel_embed[[rels[r.item()] for r in neg_triplets[:, 2]]]

    x = src_embed + rel_embed - dst_embed
    neg_norm = torch.norm(x, p=2, dim=1)

    # y = torch.ones(neg_sampling_ratio * n)
    y = torch.ones(len(pos_triplets)).to(device)

    loss_fn = nn.MarginRankingLoss(margin=5)
    loss = loss_fn(pos_norm, neg_norm, y)

    return loss


def loss_func2(triplets, neg_sampling_ratio, ent_embed, rel_embed, device='cpu'):
    """
    Triplets order: src, dst, rel
    """
    n = len(triplets)
    if type(triplets) == np.ndarray:
        triplets = torch.from_numpy(triplets)

    pos_triplets = triplets[:n // (neg_sampling_ratio + 1)]
    pos_triplets = torch.cat([pos_triplets for _ in range(neg_sampling_ratio)])

    neg_triplets = triplets[n // (neg_sampling_ratio + 1):]

    # mapping = create_mappings(pos_triplets[:, 0], pos_triplets[:, 1])
    # src_embed = ent_embed[[mapping[i.item()] for i in pos_triplets[:, 0]]]
    # dst_embed = ent_embed[[mapping[i.item()] for i in pos_triplets[:, 1]]]

    # rels = {j: i for i, j in enumerate(set([r.item() for r in pos_triplets[:, 2]]))}
    # rel_embed = rel_embed[[rels[r.item()] for r in pos_triplets[:, 2]]]

    src_embed_ = ent_embed[pos_triplets[:, 0]]
    dst_embed_ = ent_embed[pos_triplets[:, 1]]
    rel_embed_ = rel_embed[pos_triplets[:, 2]]

    x = src_embed_ + rel_embed_ - dst_embed_
    pos_norm = torch.norm(x, p=2, dim=1)

    # mapping = create_mappings(neg_triplets[:, 0], neg_triplets[:, 1])
    # src_embed = ent_embed[[mapping[i.item()] for i in neg_triplets[:, 0]]]
    # dst_embed = ent_embed[[mapping[i.item()] for i in neg_triplets[:, 1]]]

    # rels = {j: i for i, j in enumerate(set([r.item() for r in neg_triplets[:, 2]]))}
    # rel_embed = rel_embed[[rels[r.item()] for r in neg_triplets[:, 2]]]

    src_embed_ = ent_embed[neg_triplets[:, 0]]
    dst_embed_ = ent_embed[neg_triplets[:, 1]]
    rel_embed_ = rel_embed[neg_triplets[:, 2]]

    x = src_embed_ + rel_embed_ - dst_embed_
    neg_norm = torch.norm(x, p=2, dim=1)

    # y = torch.ones(neg_sampling_ratio * n)
    y = torch.ones(len(pos_triplets)).to(device)

    loss_fn = nn.MarginRankingLoss(margin=5)
    loss = loss_fn(pos_norm, neg_norm, y)

    return loss


