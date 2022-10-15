import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnns import GNN
import numpy as np
import random
from random import sample
from .Transformer import TransformerPredictor
from multiprocessing import Pool
import networkx as nx
import collections
import pdb


logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs                             

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input * ratio

        return grad_input


class TransNet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(TransNet, self).__init__()
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_sources = configs["num_sources"]
        self.h_dim = configs['feat_num']
        self.times = configs['times']
        self.hiddens = GNN(nfeat=self.h_dim,
                              nhid=configs["hidden_layers"],
                              nclass=2,    # not used
                              out_dim=configs["ndim"],
                              gnn_type=configs["type"],
                              bias=True,
                              dropout=configs["dropout"])
        self.domain_disc = torch.nn.Sequential(
            nn.Linear(configs["m_dim"], configs["m_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(configs["m_dim"], 2),
        )
        self.domain_disc_linear = torch.nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.h_dim, 2),
        )
        self.transformer = TransformerPredictor(input_dim=configs["ndim"], model_dim=configs["m_dim"], num_heads=configs["heads"],
                                        num_layers=configs["layers"], dropout=configs["dropout"])
        self.dimRedu = nn.ModuleList([torch.nn.Sequential(nn.Dropout(p=0.5), nn.Linear(ndim, self.h_dim)) for ndim in configs["input_dim"]])

        self.output_layer = nn.ModuleList([torch.nn.Sequential(
                                            nn.Dropout(p=0.5),
                                            nn.Linear(configs["m_dim"], configs["num_class"][0])) for nclass in configs["num_class"]])

        self.output_layer_ = torch.nn.Sequential(
                                    nn.Dropout(p=0.5),
                                    nn.Linear(configs["m_dim"], configs["num_class"][-1]))
        
        self.link_classifier = torch.nn.Sequential(
                                    # nn.Dropout(p=0.5),
                                    nn.Linear(configs["m_dim"] * 2, 1))
        
        self.domains = nn.ModuleList([self.domain_disc for _ in range(self.num_sources)])
        self.domains_linear = nn.ModuleList([self.domain_disc_linear for _ in range(self.num_sources)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer.apply for _ in range(self.num_sources)]
        self.t_dim = configs["ndim"]
        

    def forward(self, sinputs, tinputs, sadj, tadj, sidx, tidx, rate):
        global ratio
        ratio = rate

        s_node_num, s_seq_len, _ = sinputs[0].shape
        sh_relu = []
        sh_linear = []
        th_relu = tinputs.clone()
        for i in range(self.num_sources):
            sh_relu.append(sinputs[i].clone())


        out = []
        for i in range(self.num_sources):
            out_ = self.dimRedu[i](sh_relu[i])                 # [number_nodes, num_time, dim]
            out_ = torch.transpose(out_, 0, 1)                 # [num_time, number_nodes, dim]
            sh_linear.append(out_)
            sh_relu_ = torch.zeros([s_seq_len, s_node_num, self.t_dim]).cuda()
            for j in range(s_seq_len):
                sh_relu_[j][sidx[i][j]] = F.relu(self.hiddens(sh_linear[i][j][sidx[i][j]], sadj[i][j]))
            sh_relu[i] = self.transformer(sh_relu_.transpose(0, 1)).mean(1)
            out.append(F.log_softmax(self.output_layer[i](sh_relu[i]), dim=-1).squeeze(-1))

        
        t_node_num, t_seq_len, _ = tinputs.shape
        out_ = self.dimRedu[-1](th_relu)                     # [number_nodes, num_time, dim]
        out_ = torch.transpose(out_, 0, 1)
        th_linear = out_
        th_relu_ = torch.zeros([t_seq_len, t_node_num, self.t_dim]).cuda()
        for j in range(t_seq_len):
            th_relu_[j][tidx[j]] = F.relu(self.hiddens(th_linear[j][tidx[j]], tadj[j]))
        th_relu = self.transformer(th_relu_.transpose(0, 1)).mean(1)
        out.append(F.log_softmax(self.output_layer[0](th_relu), dim=-1).squeeze(-1))

        sdomains, tdomains, sdomains_linear, tdomains_linear = [], [], [], []
        for i in range(self.num_sources):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
            sdomains_linear.append(F.log_softmax(self.domains_linear[i](self.grls[i](sh_linear[i].reshape([s_seq_len*s_node_num, -1]))), dim=1))
            tdomains_linear.append(F.log_softmax(self.domains_linear[i](self.grls[i](th_linear.reshape([t_seq_len*t_node_num, -1]))), dim=1))
        
        return sh_relu, th_relu, sh_linear, th_linear, sdomains, tdomains, sdomains_linear, tdomains_linear, out

    def finetune(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.hiddens.parameters():
            param.requires_grad = False
        for param in self.dimRedu[-1].parameters():
            param.requires_grad = False

    def finetune_inv(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
        for param in self.hiddens.parameters():
            param.requires_grad = True
        for param in self.dimRedu[-1].parameters():
            param.requires_grad = True

    def inference(self, tinput, adj, idx, index=-1):
        t_node_num, t_seq_len, _ = tinput.shape
        h_relu = tinput.clone()
        out_ = self.dimRedu[index](h_relu)  # [number_nodes, num_time, dim]
        out_ = torch.transpose(out_, 0, 1)
        h_linear = out_
        h_relu_ = torch.zeros([t_seq_len, t_node_num, self.t_dim]).cuda()
        for j in range(t_seq_len):
            h_relu_[j][idx[j]] = F.relu(self.hiddens(h_linear[j][idx[j]], adj[j]))
        h_relu = self.transformer(h_relu_.transpose(0, 1)).mean(1)
        if index == -1:
            out = F.log_softmax(self.output_layer_(h_relu), dim=-1).squeeze(-1)
        else:
            out = F.log_softmax(self.output_layer[index](h_relu), dim=-1).squeeze(-1)

        return out, h_relu, h_linear
