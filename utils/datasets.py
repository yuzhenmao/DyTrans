import os
import networkx as nx
import pickle as pkl
import torch
import numpy as np
from collections import OrderedDict
import time
from utils.utils import *
from torch.utils.data import Dataset
from sklearn import preprocessing
import json
import scipy.sparse
from torch_geometric.io import read_txt_array
from torch_geometric.datasets import Amazon
import os.path as osp
import pdb


class GraphLoader(object):

    def __init__(self, name, root = "./data", sparse=True, args=None):

        self.name = name
        self.sparse = sparse
        if name == 'D3':
            self.path = os.path.join(root, "DBLP3.npz")
        elif name == 'D5':
            self.path = os.path.join(root, "DBLP5.npz")
        elif name == 'Br':
            self.path = os.path.join(root, "Brain.npz") 
        self._load()

    def _toTensor(self,device=None):
        for i in range(len(self.adj)):
            self.adj[i] = torch.from_numpy(self.adj[i]).cuda().float()
            self.idx[i] = torch.from_numpy(self.idx[i]).cuda().long()
        self.X = torch.from_numpy(self.X).cuda().float()
        self.Y = torch.from_numpy(self.Y).cuda().long()

    def _load(self):
        X, A, Y, idx = load_real_data(self.path)
        self.X = X
        self.Y = Y
        self.adj = A
        self.idx = idx
        # self.X = column_normalize(preprocess_features(self.X)).transpose(1, 0, 2)
        self.X = preprocess_features(self.X).transpose(1, 0, 2)
        self._toTensor()


# -----------------------------------------------------------------------------------------------------------------

class MyData:
    def __init__(self, adj, labels):
        self.adj = adj
        self.num = labels.shape[0]
        self.time = len(self.adj)
        self.labels = labels.cpu().numpy()


# -----------------------------------------------------------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data, task, ratio=2.0/3.0, few_shot=5):
        self.data = data
        self.task = task
        self.ratio = ratio
        self.few_shot = few_shot
        self.rand_index = []
        self.num = self.data.num
        self.time = self.data.time
        self.indices = list(range(int(self.num)))
        np.random.shuffle(self.indices)
        self.train_indices = self.indices[:int(self.num * self.ratio)]
        self.s_test_indices = [x for x in list(range(self.num)) if (x not in self.train_indices)]
        self.test_indices = []
        self.finetune_indices = []
        self.select_fintune()

    def __len__(self):
        if self.task == "train":
            return int(self.num * self.ratio)
        elif self.task == "test":
            if self.few_shot >= 1:
                return self.num - int(self.few_shot*len(set(self.data.labels)))
            else:
                return int(self.num * (1 - self.few_shot))
        elif self.task == "finetune":
            if self.few_shot >= 1:
                return int(self.few_shot*len(set(self.data.labels)))
            else:
                return int(self.num * self.few_shot)

    def __getitem__(self, idx):
        if self.task == "train":
            return self.train_indices[idx]
        elif self.task == "test":
            return self.test_indices[idx]
        elif self.task == "finetune":
            return self.finetune_indices[idx]

    def change_task(self, task):
        self.task = task

    def select_fintune(self):
        if self.few_shot >= 1:
            group = {}
            for i in set(self.data.labels):
                group[i] = []
                for j in range(len(self.data.labels)):
                    samp = np.random.choice(self.train_indices, 1)[0]
                    if self.data.labels[samp] == i:
                        group[i].extend([samp])
                    if len(group[i]) == self.few_shot:
                        break
            for i in set(self.data.labels):
                self.finetune_indices += group[i]
        else:
            self.finetune_indices = np.random.choice(self.train_indices, int(self.num*self.few_shot))
        self.test_indices = [x for x in list(range(self.num)) if x not in self.finetune_indices]
        np.random.shuffle(self.finetune_indices)
        

    def collate(self, batches):
        idx = [batch for batch in batches]
        idx = torch.LongTensor(idx)
        return idx