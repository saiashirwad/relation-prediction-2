from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader

from torchkge.data.DataLoader import load_fb15k237
from torchkge.data.KnowledgeGraph import KnowledgeGraph
from torchkge.models import TransEModel

from torch.utils.tensorboard import SummaryWriter

import time

from layers import MultiHeadKGAtt
from gat import KGAT
from load_data import negative_sampling, get_init_embed, get_batch_neighbors, add_inverted_triplets
from loss import loss_func, loss_func2
from eval import validate

Args = namedtuple('args', ['in_dim', 'hidden_dim', 'out_dim', 'num_heads', 'n_epochs', 'batch_size', 'lr', 'negative_rate', 'device', 'optimizer'])


def train(args: Args, kg_train, kg_val):

    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel

    writer = SummaryWriter("runs/train1")

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
            triplets, labels = negative_sampling(triplets, n_ent, args.negative_rate)
            triplets, labels = triplets.to(args.device), labels.to(args.device)

            model.zero_grad()

            # start = time.time()
            model.train()
            ent_embed_, rel_embed_ = model(triplets, ent_embed, rel_embed)
            loss = loss_func2(triplets, args.negative_rate, ent_embed_, rel_embed_, device=args.device)

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
        writer.add_scalar("Train Loss", loss, epoch)

        model.eval()
        validate(model, ent_embed, rel_embed, kg_val, 100, 'cuda')

    return loss

def train_GAT(args: Args, kg_train: KnowledgeGraph, kg_val: KnowledgeGraph):
    kg_train = add_inverted_triplets(kg_train)
    dataloader = DataLoader(kg_train, batch_size=args.batch_size, shuffle=False)


    ent_embed, rel_embed = get_init_embed()
    ent_embed, rel_embed = ent_embed.to(args.device), rel_embed.to(args.device)

    model = KGAT(args.in_dim, 50, args.out_dim, args.num_heads, kg_train.n_ent, kg_train.n_rel, args.device).to(args.device)
    # model.init_weights(*get_init_embed())

    optimizer = AdamW(model.parameters(), lr=args.lr)

    loss = 0
    model.train()
    for epoch in range(args.n_epochs):
        losses = []
        for i, batch in enumerate(dataloader):
            triplets = torch.stack(batch)
            triplets, labels = negative_sampling(triplets, kg_train.n_ent, args.negative_rate)
            triplets, labels = triplets.to(args.device), labels.to(args.device)

            src, dst = triplets[:, 0], triplets[:, 1]
            # neighbors = get_batch_neighbors(src, dst)

            model.zero_grad()
            print(f"Start: ")
            start = time.time()
            h_ent, h_rel = model(triplets, ent_embed, rel_embed)
            loss = loss_func(triplets, args.negative_rate, h_ent, h_rel, args.device)

            loss.backward()
            optimizer.step()
            print(f"Finish: {time.time() - start}")

            losses.append(loss.item())
            print(loss.item())

        loss = sum(losses) / (len(losses))
        print(f'Epoch {epoch} Loss: {loss}')

    return loss


if __name__ == '__main__':

    kg_train, kg_test, kg_val = load_fb15k237()

    args = Args(100, 200, 100, 5, 100, 2000, 0.01, 10, 'cuda', 'sgd')
    train(args, kg_train, kg_val)
