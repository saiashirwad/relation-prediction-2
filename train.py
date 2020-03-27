from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
#from torch.optim import Adam, SGD, AdamW
from torch.optim import Adam, SGD 
from torch.utils.data import DataLoader

from dataloader import load_fb15k237
from knowledgegraph import KnowledgeGraph

# from torch.utils.tensorboard import SummaryWriter

import time

from layers import MultiHeadKGAtt
from gat import KGAT
from load_data import negative_sampling, get_init_embed, get_batch_neighbors, add_inverted_triplets
from loss import loss_func, loss_func2
from eval import validate

Args = namedtuple('args', ['in_dim', 'hidden_dim', 'out_dim', 'num_heads', 'n_epochs', 'batch_size', 'lr', 'negative_rate', 'device', 'optimizer'])


def get_valid_triplets(kg_train: KnowledgeGraph, kg_test: KnowledgeGraph, kg_val: KnowledgeGraph, reverse=True):
    triplets = set()
    for kg in [kg_train, kg_test, kg_val]:
        for i in range(kg.n_facts):
            triplets.add((kg.head_idx[i], kg.tail_idx[i], kg.relations[i]))
            if reverse:
                triplets.add((kg.tail_idx[i], kg.head_idx[i], kg.relations[i]))
    print(f'Number of unique triplets: {len(triplets)}')
    return triplets



def train_kgatt(args: Args, kg_train: KnowledgeGraph, kg_test: KnowledgeGraph, kg_val: KnowledgeGraph, total_triplets=None):

    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel

    if total_triplets is None:
        total_triplets = get_valid_triplets(kg_train, kg_test, kg_val)

    dataloader = DataLoader(kg_train, batch_size=args.batch_size, shuffle=False, pin_memory=cuda.is_available())
    model = MultiHeadKGAtt(n_ent, n_rel, 100, 200, 100, args.num_heads, device=args.device).to(args.device)
    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-3)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-3)

    ent_embed, rel_embed = get_init_embed()
    ent_embed, rel_embed = ent_embed.to(args.device), rel_embed.to(args.device)

    loss = 0
    model.train()
    for epoch in range(args.n_epochs):

        losses = []
        ent_embeds = [0]
        rel_embeds = [0]

        for i, batch in enumerate(dataloader):

            triplets = torch.stack(batch)
            triplets, labels, nodes, edges = negative_sampling(triplets, n_ent, args.negative_rate)
            triplets, labels = triplets.to(args.device), labels.to(args.device)

            model.zero_grad()

            # start = time.time()
            model.train()
            ent_embed_, rel_embed_ = model(triplets, ent_embed, rel_embed, nodes, edges)
            loss = loss_func2(triplets, args.negative_rate, ent_embed_, rel_embed_, device=args.device)

            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            # print(f"Finished {time.time() - start}")

            losses.append(loss.item())

            ent_embeds[0] = ent_embed_
            rel_embeds[0] = rel_embed_

            # if i % 100 == 0:
            #     print(loss.item())
            # print(loss.item())

        loss = sum(losses) / (len(losses))
        print(f'Epoch {epoch} Loss: {loss}')
        # writer.add_scalar("Train Loss", loss, epoch)

        if epoch > 10:
            model.eval()
            validate(model, kg_val, total_triplets, 100, 'cuda')

    return loss

if __name__ == '__main__':

    kg_train, kg_test, kg_val = load_fb15k237()

    args = Args(100, 200, 100, 4, 100, 2000, 0.01, 10, 'cpu', 'sgd')
    train_kgatt(args, kg_train, kg_test, kg_val)
