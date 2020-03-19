import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchkge.data.DataLoader import load_fb15k237

def validate(model, dataset='fb15k237'):
    if dataset == 'fb15k237':
        _, _, kg = load_fb15k237()
    else:
        print("Dataset not suppported")
        return 
    
    
