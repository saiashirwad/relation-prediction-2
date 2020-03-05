import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchkge.data.DataLoader import *
from torch import cuda 


from kgatt import * 
from load_data import *


batch_size = 1024
n_epochs = 5

n_ent, n_rel = kg_train 
negative_rate = 10

kg_train, kg_test, kg_val = load_fb15k237()
dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, pin_memory=cuda.is_available())

ent_embed, rel_embed = get_init_embed()

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        triplets = torch.stack(batch).t() 
        triplets, labels = negative_sampling(triplets, n_ent, 10)

