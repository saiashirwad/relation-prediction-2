import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchkge.data.DataLoader import load_fb15k237
from torchkge.data.KnowledgeGraph import KnowledgeGraph


def validate(model: nn.Module, ent_embed, rel_embed, kg: KnowledgeGraph, batch_size=1000, device='cuda'):
    batch_size = 1
    dataloader = DataLoader(kg, batch_size, shuffle=False)

    hits = [[] for _ in range(10)]
    ranks = [] 

    rankHs, rankTs = [], []

    # model.eval()
    for i, batch in enumerate(dataloader):
        if i > 200:
            break
        src, dst, rel = batch

        triplets = torch.stack(batch).t().to(device)

        ent_embed_, rel_embed_ = model(triplets, ent_embed, rel_embed)

        if device == 'cuda':
            src = src.to(device)
            dst = dst.to(device)
            rel = rel.to(device)
        
        src = ent_embed_[src]
        dst = ent_embed_[dst]
        rel = rel_embed_[rel]

        loss = torch.norm(src + rel - dst, 2, 1).repeat(kg.n_ent)

        src_ = src.repeat(kg.n_ent, 1)
        dst_ = dst.repeat(kg.n_ent, 1)
        rel_ = rel.repeat(kg.n_ent, 1)

        ent_embed_ = ent_embed_.repeat(batch_size, 1)

        loss_h = torch.norm(ent_embed_ + rel_ - dst_, 2, 1)
        loss_t = torch.norm(src_ + rel_ - ent_embed_, 2, 1)

        rankH = torch.nonzero(F.relu(loss - loss_h)).shape[0]
        rankT = torch.nonzero(F.relu(loss - loss_t)).shape[0]

        rankHs.append(rankH)
        rankTs.append(rankT)
    
    # print(sum(rankHs) / len(rankHs))
    # print(sum(rankTs) / len(rankTs))
    v = (sum(rankHs)/len(rankHs)) + (sum(rankTs)/len(rankTs))
    print(v/2)

        

# def validate(model: nn.Module, embeddings: List[nn.Module], kg: KnowledgeGraph):
#     hits_at_1 = 0.0 
#     hits_at_3 = 0.0 
#     hits_at_10 = 0.0 
#     mrr = 0.0 

#     dataloader = DataLoader(kg, 10)
#     for i, batch in enumerate(dataloader): 
