import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchkge.data.DataLoader import load_fb15k237
from torchkge.data.KnowledgeGraph import KnowledgeGraph

import multiprocessing as mp 

from typing import Set, Tuple


def validate(model: nn.Module, ent_embed, rel_embed, kg: KnowledgeGraph, total_triplets: Set[Tuple], batch_size=1000, device='cuda'):
    batch_size = 1


    n_ranks = 10
    k = kg.n_ent

    dataloader = DataLoader(kg, batch_size, shuffle=False)

    hits = [[] for _ in range(10)]
    ranks = [] 

    rankHs, rankTs = [], []

    head_rank_mean, tail_rank_mean, filtered_head_rank_mean, filtered_tail_rank_mean = [0] * 4

    head_hits_10_raw, head_hits_10_filter, tail_hits_10_raw, tail_hits_10_filter = [0] * 4

    # model.eval()
    for i, batch in enumerate(dataloader):
        if i > 200:
            break

        h = batch[0]
        t = batch[1]
        r = batch[2]
        batch[0] = torch.cat([h, t])
        batch[1] = torch.cat([t, h])
        batch[2] = torch.cat([r, r])

        
        src, dst, rel = batch

        triplets = torch.stack(batch).t().to(device)

        # ent_embed_, rel_embed_ = model(triplets, ent_embed, rel_embed)
        
        #Eh
        ent_embed_, rel_embed_ = ent_embed, rel_embed

        if device == 'cuda':
            src = src.to(device)
            dst = dst.to(device)
            rel = rel.to(device)
        
        src = ent_embed_[src]
        dst = ent_embed_[dst]
        rel = rel_embed_[rel]

        loss = torch.norm(src + rel - dst, 2, 1)
        loss = loss.repeat(kg.n_ent)

        src_ = src.repeat(kg.n_ent, 1)
        dst_ = dst.repeat(kg.n_ent, 1)
        rel_ = rel.repeat(kg.n_ent, 1)

        ent_embed_ = ent_embed_.repeat(2 * batch_size, 1)

        dist_head_prediction = ent_embed_ + rel_ - dst_
        dist_tail_prediction = src_ + rel_ - ent_embed_

        _, head_prediction = torch.topk(torch.sum(torch.abs(dist_head_prediction), dim=1), k=k)
        _, tail_prediction = torch.topk(torch.sum(torch.abs(dist_tail_prediction), dim=1), k=k)

        head_prediction = head_prediction.cpu()
        tail_prediction = tail_prediction.cpu()

        # eval_queue = mp.JoinableQueue()
        # rank_queue = mp.Queue()

        # for i in n_ranks:
        # start process

        # To go inside calc_rank!! 
        head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = [0] * 4


        s, d, r = [b[0] for b in batch] # Ugh

        for candidate in head_prediction:
            if candidate == s:
                break
            else:
                head_rank_raw += 1 
                if (candidate, d, r) in total_triplets:
                    continue
                else:
                    head_rank_filter += 1
        
        for candidate in tail_prediction:
            if candidate == d:
                break 
            else:
                tail_rank_raw += 1
                if (s, candidate, r) in total_triplets:
                    continue
                else:
                    tail_rank_filter += 1 
        
        # head_ranks.append(head_rank_raw)
        # tail_ranks.append(tail_rank_raw)
        # filtered_head_ranks.append(head_rank_filter)
        # filtered_tail_ranks.append(tail_rank_filter)

        head_rank_mean += head_rank_raw
        tail_rank_mean += tail_rank_raw

        filtered_head_rank_mean += head_rank_filter
        filtered_tail_rank_mean += tail_rank_filter

        if head_rank_raw < 10:
            head_hits_10_raw += 1
        
        if tail_rank_raw < 10:
            tail_hits_10_raw += 1

        if head_rank_filter < 10:
            head_hits_10_filter += 1
        
        if tail_rank_filter < 10:
            tail_hits_10_filter += 1
        
    
    head_rank_mean /= 200
    tail_rank_mean /= 200

    filtered_head_rank_mean /= 200
    filtered_tail_rank_mean /= 200

    print(f'Head Rank Mean : {head_rank_mean} | Hits@10 : {head_hits_10_raw}')
    print(f'Tail Rank Mean : {tail_rank_mean} | Hits@10 : {tail_hits_10_raw}')

    print(f'Filtered Head Rank Mean: {filtered_head_rank_mean}')
    print(f'Filtered Tail Rank MEan: {filtered_tail_rank_mean}')


    print()

        



                


def calc_rank(in_queue: mp.JoinableQueue, out_queue):
    """
    Adapted from https://github.com/ZichaoHuang/TransE/
    in_queue: triplet, head_prediction, tail_prediction
    """
    while True:
        predictions = in_queue.get()
        if predictions is None:
            in_queue.task_done()
            return 
        else:
            triplet, head_prediction, tail_prediction = predictions
            src, dst, rel = triplet

            src_rank_raw, dst_rank_raw, src_rank_filter, dst_rank_filter = [0] * 4
            pass 