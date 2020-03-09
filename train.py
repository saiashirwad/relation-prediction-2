from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchkge.data.DataLoader import load_fb15k237

from layers import MultiHeadKGAtt
from load_data import negative_sampling, get_init_embed
from loss import loss_func

Args = namedtuple('args', ['in_dim', 'hidden_dim', 'out_dim', 'n_epochs', 'batch_size', 'lr', 'negative_rate', 'device'])


def train(args: Args, kg_train, kg_val):

    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel

    dataloader = DataLoader(kg_train, batch_size=args.batch_size, shuffle=False, pin_memory=cuda.is_available())
    model = MultiHeadKGAtt(n_ent, n_rel, 100, 200, 100, 10, device=args.device).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    ent_embed, rel_embed = get_init_embed()
    ent_embed, rel_embed = ent_embed.to(args.device), rel_embed.to(args.device)

    for epoch in range(args.n_epochs):

        losses = []

        for i, batch in enumerate(dataloader):
            triplets = torch.stack(batch)
            triplets, labels = negative_sampling(triplets, n_ent, args.negative_rate)
            triplets, labels = triplets.to(args.device), labels.to(args.device)

            model.zero_grad()

            ent_embed_, rel_embed_ = model(triplets, ent_embed, rel_embed)
            loss = loss_func(triplets, args.negative_rate, ent_embed_, rel_embed_, device=args.device)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        loss = sum(losses) / (len(losses))
        print(f'Epoch {epoch} Loss: {loss}')


if __name__ == '__main__':

    kg_train, kg_test, kg_val = load_fb15k237()

    args = Args(100, 200, 100, 10, 1024, 0.01, 10, 'cpu')
    train(args, kg_train, kg_val)
