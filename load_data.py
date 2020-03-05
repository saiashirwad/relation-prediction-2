from torchkge.data.DataLoader import *
import numpy as np 

def text2vec(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [[float(i) for i in line.split()] for line in lines]
    
    t = torch.empty((len(lines), len(lines[0])))
    for i in range(len(lines)):
        for j in range(len(lines[0])):
            t[i][j] = lines[i][j]
    
    return t.type(torch.FloatTensor) 


def text2triple(file, node2id, edge2id):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [[i for i in line.split()] for line in lines]

    triples = []
    for line in lines:
        triples.append((
            node2id[line[0]],
            edge2id[line[1]],
            node2id[line[2]]
        ))
    
    return triples

def text2dict(file):
    dict = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()
            dict[temp[0]] = int(temp[1])
    
    return dict 

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def get_init_embed(datafolder='data/FB15k-237/'):
    ent_embed = np.loadtxt(f'{datafolder}entity2vec.txt')
    rel_embed = np.loadtxt(f'{datafolder}relation2vec.txt')

