import importlib

from layers import SpecialSpmmFinal

import train 
importlib.reload(train)
from train import *

import torch
import torch.nn as nn 

from functools import reduce
from operator import mul

from torch_scatter import scatter
from embedding import EmbeddingMul2

kg_train, kg_test, kg_val = load_fb15k237()
args = Args(100, 200, 100, 2, 100, 1000, 0.001, 10, 'cuda', 'sgd')

n_ent, n_rel = kg_train.n_ent, kg_train.n_rel
total_triplets = get_valid_triplets(kg_train, kg_test, kg_val)

dataloader = DataLoader(kg_train, batch_size=args.batch_size, shuffle=False, pin_memory=cuda.is_available())
ent_embed, rel_embed = get_init_embed()

class KGLayer(nn.Module):
    def __init__(self, n_entities, n_relations, ent_embed, rel_embed, in_dim, out_dim, concat=True, device="gpu"):
        super(KGLayer, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.a = nn.Linear(3 * in_dim, out_dim).to(device)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.concat = concat

        self.a_2 = nn.Linear(out_dim, 1).to(device)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

        self.special_spmm_final = SpecialSpmmFinal()

        # self.ent_embed = EmbeddingMul2(n_entities, in_dim, ent_embed, True, torch.device(device))
        # self.rel_embed = EmbeddingMul2(n_relations, in_dim, rel_embed, True, torch.device(device))

        self.ent_embed = nn.Embedding(n_entities, in_dim).to(device)
        self.rel_embed = nn.Embedding(n_relations, in_dim).to(device)
    
    def forward(self, triplets):
        N = self.n_entities

        h = torch.cat((
            self.ent_embed(triplets[:, 0]),
            self.ent_embed(triplets[:, 1]),
            self.rel_embed(triplets[:, 2])
        ), dim=1)
        c = self.a(h)
        b = F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[1]])

        e_b_sum = self.special_spmm_final(edges, e_b, N, e_b.shape[0], 1)
        temp1 = e_b * c

        h_sum = self.special_spmm_final(edges, temp1,  N, e_b.shape[0], self.out_dim)

        hs = h_sum
        ebs = e_b_sum
        ebs[ebs == 0] = 1e-12


        h_ent = hs / ebs

        index = triplets[:, 2]
        h_rel = scatter(temp1, index=index, dim=0, reduce="mean")

        del h 
        torch.cuda.empty_cache()

        if self.concat:
            return F.elu(h_ent), F.elu(h_rel)
        else:
            return h_ent, h_rel

model = KGLayer(n_ent, n_rel, ent_embed, rel_embed, 100, 100, True, "cuda")

optimizer = SGD(model.parameters(), lr=args.lr)

batches = [b for b in dataloader]


for epoch in range(20):
    losses = []
    for batch in batches:
        triplets = torch.stack(batch)
        triplets, labels, nodes, edges = negative_sampling(triplets, n_ent, args.negative_rate)
        triplets, labels = triplets.to(args.device), labels.to(args.device)

        model.zero_grad()

        # start = time.time()
        model.train()
        ent_embed_, rel_embed_ = model(triplets)
        loss = loss_func2(triplets, args.negative_rate, ent_embed_, rel_embed_, device="cuda")
        loss.backward()
        optimizer.step()

        # del triplets
        # del labels
        # del ent_embed_
        # del rel_embed_
        # del nodes 
        # del edges  
        torch.cuda.empty_cache()

        losses.append(loss.item())
        # print(loss.item())
        # del loss 
        torch.cuda.empty_cache()

    print(sum(losses) / len(losses))
    print()