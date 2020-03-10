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
        g[head].append(tail)

    return g

def get_batch_neighbors(src: torch.Tensor, dst: torch.Tensor) -> Dict[int, List[int]]:
    return generate_graph(src, dst)