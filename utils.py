import torch

from typing import List, Dict

def create_mappings(src: torch.Tensor, dst:torch.Tensor) -> Dict[int, List[int]]:
    src = [s.item() for s in src]
    dst = [d.item() for d in dst]

    nodes = {j : i for i, j in enumerate(sorted(list(set(src + dst))))}

    return nodes


def generate_graph(src, dst):
    if type(src) == torch.Tensor:
        src = [s.item() for s in src]
        dst = [d.item() for d in dst]

    g = {}
    for head, tail in zip(src, dst):
        if head not in g.keys():
            g[head] = []
        if tail not in g.keys():
            g[tail] = []
        g[head].append(tail)
        g[tail].append(head)

    return {i: list(set(g[i])) for i in g.keys()}

def get_batch_neighbors(src: torch.Tensor, dst: torch.Tensor) -> Dict[int, List[int]]:
    return generate_graph(src, dst)

def rel2edge(src, dst, rel):
    src = [s.item() for s in src]
    dst = [d.item() for d in dst]
    rel = [r.item() for r in rel]

    r2e = {}
    for i, r in enumerate(rel):
        if r not in r2e.keys():
            r2e[r] = []

        r2e[r].append((src[i], dst[i]))
        r2e[r].append((dst[i], src[i]))

    return {i: list(set(r2e[i])) for i in r2e.keys()}
