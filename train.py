import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchkge.data.DataLoader import load_fb15k237
from torch import cuda
from torch.optim import Adam

from layers import MultiHeadKGAtt

from load_data import negative_sampling, get_init_embed

from sampler import ModifiedPositionalNegativeSampler

from collections import namedtuple

from loss import loss_func


batch_size = 1024
n_epochs = 5


negative_rate = 10

kg_train, kg_test, kg_val = load_fb15k237()
dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, pin_memory=cuda.is_available())

sampler = ModifiedPositionalNegativeSampler(kg_train, kg_test=kg_test)
n_ent, n_rel = kg_train.n_ent, kg_train.n_rel

ent_embed, rel_embed = get_init_embed()
# model = KGAtt(n_ent, n_rel, 100, 100, 0)
# model = MultiHeadKGAtt(n_ent, n_rel, 100, 200, 100, 10)

Args = namedtuple('args', ['in_dim', 'hidden_dim', 'out_dim', 'n_epochs', 'batch_size', 'lr'])
args = Args(100, 200, 100, 10, 1024, 0.01)

def train(args: Args):
    global ent_embed
    global rel_embed
    model = MultiHeadKGAtt(n_ent, n_rel, 100, 200, 100, 10)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):

        losses = []

        for i, batch in enumerate(dataloader):
            triplets = torch.stack(batch)
            triplets, labels = negative_sampling(triplets, n_ent, negative_rate)

            model.zero_grad()

            ent_embed_, rel_embed_ = model(triplets, ent_embed, rel_embed)
            loss = loss_func(triplets, negative_rate, ent_embed_, rel_embed_)

            loss.backward()
            optimizer.step()

            losses.append(loss)

        losses = sum(losses) / (len(losses) * args.batch_size)


train(args)
