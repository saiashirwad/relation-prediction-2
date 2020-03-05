import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchkge.data.DataLoader import *
from torch import cuda 

from layers import *

from load_data import *


batch_size = 1024
n_epochs = 5


negative_rate = 10

kg_train, kg_test, kg_val = load_fb15k237()
dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, pin_memory=cuda.is_available())

n_ent, n_rel = kg_train.n_ent, kg_train.n_rel

ent_embed, rel_embed = get_init_embed()
model = KGAtt(n_ent, n_rel, 100, 100, 0) 

for epoch in range(1):
    for i, batch in enumerate(dataloader):
        triplets = torch.stack(batch).t() 
        # triplets, labels = negative_sampling(triplets, n_ent, negative_rate)

        out = model(triplets, ent_embed, rel_embed)


