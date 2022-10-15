import numpy as np
import scipy.sparse as sp
import pandas as pd
import warnings
from sklearn.metrics import f1_score
import torch
import math
import sys
import logging
import networkx as nx
from tqdm import tqdm
import os
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
import pymetis
import torch.nn.functional as F
import collections
from random import sample
from queue import Queue
import heapq
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from torch.distributions import Categorical
import scipy
import copy


def load_real_data(dataset_name):
    dataset = np.load(dataset_name)
    
    Adj = np.array(dataset['adjs'])    #(n_time, n_node, n_node)

    temp = Adj.sum(0).sum(1)
    non_zero_index = np.nonzero(temp)[0]
    Adj = Adj[:,non_zero_index,:]
    Adj = Adj[:,:,non_zero_index]

    Labels = np.array(np.argmax(dataset['labels'], axis=1), dtype=np.long)  #(n_node, num_classes)
    Features = np.array(dataset['attmats'])                                 #(n_node, n_time, att_dim)

    Features = Features[non_zero_index]
    Labels = Labels[non_zero_index]
    node_lst = []
    Adj_ = []
    for i in range(Adj.shape[0]):
        node_lst.append(np.arange(Adj.shape[1]))
        Adj_.append(Adj[i])
    
    return Features.transpose(1, 0, 2), Adj_, Labels, node_lst


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            rowsum = np.array(features[i, j].sum(-1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = np.diag(r_inv)
            features[i, j] = r_mat_inv.dot(features[i, j].reshape(1, -1))
    return features


def column_normalize(tens):
    for i in range(tens.shape[0]):
        tens[i] = tens[i] - tens[i].mean(axis=0)
    return tens


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.INFO)
    logger.addHandler(std_handler)
    return logger


def viz_single(tvec, rootdir, tname, epoch, loss, label, few_shot, adapt=0, sudo_label=None, ass_node=None):

    if not os.path.exists(os.path.join(rootdir, tname)):
        os.makedirs(os.path.join(rootdir, tname))

    def discrete_cmap(n, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        base = plt.cm.get_cmap(base_cmap)
        return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)

    color = ['red']
    vec = []
    legend = []
    tvec_ = tvec.cpu().tolist()
    vec += tvec_
    
    vec = np.asarray(vec)
    label = label.cpu().numpy()
    few_shot = few_shot.cpu().numpy()
    vec_ = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vec)

    if sudo_label is not None:
        sudo_label = sudo_label.cpu().numpy()
        if np.max(sudo_label)-np.min(sudo_label) > 0:
            plt.figure(figsize=(10, 10))
            sc = plt.scatter(vec_[:,0], vec_[:,1], alpha=0.7, cmap=discrete_cmap(np.max(sudo_label)-np.min(sudo_label)+1, 'jet'), c=sudo_label, edgecolors='black', vmin = np.min(sudo_label)-.5, vmax = np.max(sudo_label)+.5)
            cax = plt.colorbar(sc, ticks=np.arange(np.min(sudo_label),np.max(sudo_label)+1))
            plt.grid()
            if adapt == 0:
                plt.title(f'acc is {loss}')
                logdir = os.path.join(rootdir, tname, 'sudo_finetune_full_'+str(epoch+1)+'.png')
            elif adapt == 1:
                # plt.title(f'loss is {loss}')
                logdir = os.path.join(rootdir, tname, 'sudo_pretrain_latent_'+str(epoch)+'.png')

            plt.axis('off')
            plt.savefig(logdir)
            plt.close()

    if ass_node is not None:
        ass_node = ass_node.cpu().numpy()
        plt.figure(figsize=(10, 10))
        sub_label = np.where(ass_node==-1)
        other_label = np.where(ass_node!=-1)
        plt.scatter(vec_[:,0][sub_label], vec_[:,1][sub_label], alpha=0.2, c='white', edgecolors='black')
        sc = plt.scatter(vec_[:,0][other_label], vec_[:,1][other_label], alpha=0.9, cmap=discrete_cmap(np.max(label)+1, 'jet'), c=label[other_label], edgecolors='black', vmin = -.5, vmax = np.max(label)+.5)
        cax = plt.colorbar(sc, ticks=np.arange(np.min(label),np.max(label)+1))
        plt.grid()
        if adapt == 0:
            logdir = os.path.join(rootdir, tname, 'finetune_ass'+str(epoch+1)+'.png')
        elif adapt == 1:
            logdir = os.path.join(rootdir, tname, 'pretrain_ass_'+str(epoch)+'.png')
        plt.axis('off')
        plt.savefig(logdir)
        plt.close()

    plt.figure(figsize=(10, 10))
    sub_label = np.where(few_shot==-1)
    other_label = np.where(few_shot!=-1)
    plt.scatter(vec_[:,0][sub_label], vec_[:,1][sub_label], alpha=0.2, c='white', edgecolors='black')
    sc = plt.scatter(vec_[:,0][other_label], vec_[:,1][other_label], alpha=0.9, cmap=discrete_cmap(np.max(label)+1, 'jet'), c=label[other_label], edgecolors='black', vmin = -.5, vmax = np.max(label)+.5)
    cax = plt.colorbar(sc, ticks=np.arange(np.min(label),np.max(label)+1))
    plt.grid()
    if adapt == 0:
        logdir = os.path.join(rootdir, tname, 'finetune_fewshot'+str(epoch+1)+'.png')
    elif adapt == 1:
        logdir = os.path.join(rootdir, tname, 'pretrain_fewshot_'+str(epoch)+'.png')
    plt.axis('off')
    plt.savefig(logdir)
    plt.close()

    plt.figure(figsize=(10, 10))
    sc = plt.scatter(vec_[:,0], vec_[:,1], alpha=0.7, cmap=discrete_cmap(np.max(label)-np.min(label)+1, 'jet'), c=label, edgecolors='black', vmin = np.min(label)-.5, vmax = np.max(label)+.5)
    cax = plt.colorbar(sc, ticks=np.arange(np.min(label),np.max(label)+1))
    plt.grid()
    if adapt == 0:
        plt.title(f'acc is {loss}')
        logdir = os.path.join(rootdir, tname, 'finetune_full_'+str(epoch+1)+'.png')
    elif adapt == 1:
        # plt.title(f'loss is {loss}')
        logdir = os.path.join(rootdir, tname, 'pretrain_latent_'+str(epoch)+'.png')
    plt.axis('off')
    plt.savefig(logdir)
    plt.close()


def assign_sudo_label(logprobs, labels, device, dicts = None, gcn_labels=None, all_labels=None, few_shot_idx=None, relax=False):
    temp_l = labels.clone().cpu().numpy()
    temp_p = logprobs.detach().clone().cpu().numpy()
    n = temp_l.max() + 1
    m = temp_p.shape[-1]
    few_shot = temp_p.shape[0] // n
    clusters = {i: [] for i in range(n)}
    for idx, label in enumerate(temp_l):
        clusters[label].append(idx)

    if dicts is None:
        means = {}
        dicts = {}
        dicts_inv = {}
        for i in set(temp_l):
            means[i] = temp_p[clusters[i]].max(0)*0
            for j in range(m):
                means[i][j] = np.sum(np.argmax(temp_p[clusters[i]], 1)==j) / np.sqrt(np.sum(temp_l==i))
        
        q_t = Queue(maxsize = n)
        set_s = set()
        for i in range(m):
            set_s.add(i)
        for i in set(temp_l):
            q_t.put(i)

        if relax is False:
            while not q_t.empty():
                if len(set_s) == 0:
                    for i in set(temp_l):
                        means[i] = temp_p[clusters[i]].max(0)*0
                        for j in range(m):
                            means[i][j] = np.sum(np.argmax(temp_p[clusters[i]], 1)==j) / np.sqrt(np.sum(temp_l==i))
                    break
                lll = q_t.get()
                max_i = np.argmax(means[lll])
                max_p = np.max(means[lll])
                if max_i in set_s:
                    set_s.remove(max_i)
                    dicts[lll] = max_i
                    dicts_inv[max_i] = lll
                elif means[dicts_inv[max_i]][max_i] < max_p:
                    q_t.put(dicts_inv[max_i])
                    means[dicts_inv[max_i]][max_i] = np.min(means[dicts_inv[max_i]]) - 1
                    dicts[lll] = max_i
                    dicts_inv[max_i] = lll
                else:
                    q_t.put(lll)
                    means[lll][max_i] = np.min(means[lll]) - 1

        while not q_t.empty():
            lll = int(q_t.get())
            max_i = np.argmax(means[lll])
            dicts[lll] = int(max_i)
    
    new_label = []
    for label in temp_l:
        new_label.append(dicts.get(label, 0))

    return torch.LongTensor(new_label).to(device), dicts