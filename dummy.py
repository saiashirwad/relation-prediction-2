# %%
import train 
from train import * 

import utils
import layers 
import loss
import eval

from importlib import reload
# %%

kg_train, kg_test, kg_val = load_fb15k237()
valid_triplets = get_valid_triplets(kg_train, kg_test, kg_val)
# %%
#('args', ['in_dim', 'hidden_dim', 'out_dim', 'num_heads', 'n_epochs', 'batch_size', 'lr', 'negative_rate', 'device', 'optimizer'])
args = Args(100, 200, 100, 4, 2, 1000, 0.01, 10, 'cuda', 'sgd')
# %%

reload(utils)
reload(layers)
reload(loss)
reload(eval)
reload(train)
from train import * 
train_kgatt(args, kg_train, kg_test, kg_val, valid_triplets)

# %%
