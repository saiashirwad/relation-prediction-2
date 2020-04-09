import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math
from torch.utils.data import DataLoader
import multiprocessing
import time 
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np 

def evaluate(triplets, total_triplets, model, ent2eval="head", norm=1):
    
    ent_embed = model.ent_embed.weight.numpy()
    rel_embed = model.rel_embed.weight.numpy()

    src_ = [triplet[0] for triplet in triplets]
    dst_ = [triplet[1] for triplet in triplets]
    rel_ = [triplet[2] for triplet in triplets]

    src = model.ent_embed(src_)
    dst = model.ent_embed(dst_)
    rel = model.rel_embed(rel_)

    if ent2eval == "head":
        if norm == 1:
            dist = pairwise_distances(dst - rel, ent_embed, metric="manhattan")
        else:
            dist = pairwise_distances(dst - rel, rel_embed, metric="euclidean")
        
        rankArrayHead = np.argsort(dist, axis=1)

        # do filtered later
        rankListHead = [int(np.argwhere(e[1] == e[0])) for e in zip(src_, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListHead)
        hit10count = len(isHit10ListHead)
        tripleCount = len(rankListHead)

    

class EvalProcess(multiprocessing.Process):
    def __init__(self, L, total_triplets, model, queue=None):
        super(EvalProcess, self).__init__()

        self.L = L 
        self.queue = queue
        self.total_triplets = total_triplets
        self.model = model 
    
    def run(self):
        while True:
            testList = self.queue.get()
            try:
                self.process_data(testList)
            except:
                time.sleep(5)
                self.process_data(testList)
            
            self.queue.task_done()
        
    
    def process_data(self, testList):
        pass 

def evaluation(model, kg, batch_size, total_triplets, num_processes):
    dataloader = DataLoader(kg, 1, shuffle=True)
    data = [d for d in dataloader]

    len_split = math.ceil(len(data) / num_processes)
    data_split = [data[i: i + len_split] for i in range(0, len(data), len_split)]

    with multiprocessing.Manager as manager:
        L = manager.list() 
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = EvalProcess(L, total_triplets, model)
            workerList.append(worker)
            worker.start()
        
        for split in data_split:
            queue.put(split)
        
        queue.join()

        result = list(L)

        for worker in workerList:
            worker.terminate()

