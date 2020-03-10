# from torchkge.data.DataLoader import *
import numpy as np
import torch
from torchkge.data.DataLoader import load_fb15k237
from torchkge.data.KnowledgeGraph import KnowledgeGraph

from typing import Dict, List

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
    if pos_samples.shape[0] == 3:
        pos_samples = pos_samples.t()
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    # print(neg_samples.shape, "negative_samples")
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 1] = values[obj]

    return torch.from_numpy(np.concatenate((pos_samples, neg_samples))), torch.from_numpy(labels)

def get_init_embed(datafolder='data/FB15k-237/'):
    ent_embed = np.loadtxt(f'{datafolder}entity2vec.txt')
    rel_embed = np.loadtxt(f'{datafolder}relation2vec.txt')

    return torch.from_numpy(ent_embed).to(torch.float32), torch.from_numpy(rel_embed).to(torch.float32)

def generate_graph(src, dst):
    if type(src) == torch.Tensor:
        src = [s.item() for s in src]
        dst = [d.item() for d in dst]

    g = {}
    for head, tail in zip(src, dst):
        if head not in g.keys():
            g[head] = []
        g[head].append(tail)

    return g

def get_batch_neighbors(src: torch.Tensor, dst: torch.Tensor) -> Dict[int, List[int]]:
    return generate_graph(src, dst)

def add_inverted_triplets(kg: KnowledgeGraph) -> KnowledgeGraph:
    src, dst, rel = kg.head_idx, kg.tail_idx, kg.relations
    kg.head_idx = torch.cat((src, dst))
    kg.tail_idx = torch.cat((dst, src))
    kg.relations = torch.cat((rel, rel))
    kg.n_facts *= 2

    return kg

def create_mappings(src, dst):
    src = [s.item() for s in src]
    dst = [d.item() for d in dst]

    nodes = sorted(list(set(src + dst)))

    return nodes