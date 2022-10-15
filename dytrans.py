import numpy as np
import time
import argparse
import pickle
import torch
import torch.optim as optim
import networkx as nx
import torch.nn.functional as F
from model.model import *
import matplotlib.pyplot as plt
from utils.utils import *
from utils.datasets import *
from torch.utils.data import DataLoader
from utils.logger import Logger
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import metrics
import warnings
import os
import json
from tqdm import tqdm
import sys
import pdb
import copy


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="logs")
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=93)
parser.add_argument("-u", "--mu", help="coefficient for the domain adversarial loss", type=float, default=1e-2)
parser.add_argument('--hidden', type=list, default=[64, 32], help='Number of hidden units.')  # append to the last layer
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=16)
parser.add_argument("--datasets",type=str, default="D3+D5")
parser.add_argument('--dim', type=int, default=16, help='Number of output dim.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--finetune_epoch', type=int, default=500, help='Finetune Epoch.')
parser.add_argument('--finetune_lr', type=float, default=1e-3, help='Finetune learning rate.')
parser.add_argument('--weight', action='append', type=str, default=None, help='trained model weight')
parser.add_argument("--ratio", type=float, default=1.0)   # train dataset / all dataset
parser.add_argument("--root_dir", type=str, default='logs')   # output dir path
parser.add_argument("--time", type=bool, default=True)   # add time to store path
parser.add_argument("--save", type=bool, default=False)   # save the model or not
parser.add_argument("--plt_i", type=int, default=1000)   # plot interval
parser.add_argument("--few_shot", type=float, default=5)  # few_shot number (support decimal and int)
parser.add_argument("--only", type=bool, default=False)   # only use the first dataset as target
parser.add_argument("--viz", type=bool, default=False)   # visualization
parser.add_argument("--gnn", type=str, default="gcn")   # choose to use which gnn: [gin, gcn, gat, graphsage]
parser.add_argument("--disc", type=str, default='3')   # domain discriminator type
parser.add_argument("--pre_finetune", type=int, default=0)   # fix diRedu and hidden, finetune classfier
parser.add_argument("--_alpha", action='append', type=str, default=None)   # weight for target domain
parser.add_argument("--feat_num", type=int, default=128)   # number of MLP output dimension
parser.add_argument("--layers", type=int, default=1)   # number of attention layers
parser.add_argument("--heads", type=int, default=4)   # number of attention heads
parser.add_argument("--m_dim", type=int, default=16)   # number of attention features


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logger = get_logger(args.name)
log = Logger('TransNet', str(args.epoch) + '-' + str(args.finetune_epoch) + '-' + str(args.few_shot) +
             '-' + args.datasets + '-' + str(args.mu), root_dir=args.root_dir, with_timestamp=args.time)

log.add_params(vars(args))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def train_epoch(model, dataset, dataloader, optimizer, num_data_sets, args, source_insts, 
                target_insts, source_adj, source_labels, source_idx, target_labels, target_adj, target_idx, 
                i, device, rate, sudo_weight, nclass, epoch):
    few_shot_idx = dataset[i].finetune_indices
    dloaders = []
    model.train()
    for j in range(num_data_sets):
        dataset[j].change_task('train')
        dloaders.append(iter(dataloader[j]))

    for j in range(num_data_sets):
        if j != i:
            s_labels = torch.zeros(dataset[j].num * dataset[j].time, requires_grad=False).type(
                torch.LongTensor).to(device)
    t_labels = torch.ones(dataset[i].num * dataset[i].time, requires_grad=False).type(
        torch.LongTensor).to(device)

    for j in range(num_data_sets):
        if j != i:
            s_labels_ = torch.zeros(dataset[j].num, requires_grad=False).type(
                torch.LongTensor).to(device)
    t_labels_ = torch.ones(dataset[i].num, requires_grad=False).type(
        torch.LongTensor).to(device)

    optimizer.zero_grad()

    sh_relu, th_relu, sh_linear, th_linear, sdomains, tdomains, sdomains_0, tdomains_0, out = \
        model(source_insts, target_insts, source_adj, target_adj, source_idx, target_idx, rate)
    
    sudo_labels, _ = assign_sudo_label(out[-1][few_shot_idx], target_labels[few_shot_idx], device, dicts=None)

    dt_losses = F.nll_loss(out[0].reshape(-1, nclass[0]), source_labels[0].reshape(-1)) + sudo_weight * F.nll_loss(out[-1][few_shot_idx].reshape(-1, nclass[0]), sudo_labels)

    # Domain loss                                    
    domain_losses = torch.stack([F.nll_loss(sdomains[j], s_labels_) +
                                F.nll_loss(tdomains[j], t_labels_)
                                for j in range(num_sources)])
    domain_losses_0 = torch.stack([F.nll_loss(sdomains_0[j], s_labels) +
                                F.nll_loss(tdomains_0[j], t_labels)
                                for j in range(num_sources)])
    if args.disc == '2':
        domain_losses = domain_losses_0
    if args.disc == '3':
        domain_losses = domain_losses + 5*domain_losses_0

    loss = torch.max(dt_losses) + args.mu * torch.min(domain_losses)

    running_loss = loss.item()
    dt_loss = torch.max(dt_losses).item()
    domain_loss = torch.max(domain_losses).item()

    loss.backward()
    optimizer.step()

    return running_loss, dt_loss, domain_loss, sh_relu, th_relu, sh_linear, th_linear


def test(model, dataset, args, insts, idx, labels, adj, device, index=-1):
    model.eval()
    _insts = insts.clone().detach().to(device)
    _labels = labels.clone().detach().cpu()
    with torch.no_grad():
        logprobs, vec, linear = model.inference(_insts, adj, idx, index)
    preds_labels = torch.max(logprobs[dataset.test_indices], 1)[1].squeeze_().cpu()
    pred_acc = torch.sum(preds_labels == _labels[dataset.test_indices]).item() / float(len(dataset.test_indices))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mic,mac=f1_score(_labels[dataset.test_indices], preds_labels, average="micro"), f1_score(_labels[dataset.test_indices], preds_labels, average="macro")
        matrix = confusion_matrix(_labels[dataset.test_indices], preds_labels)
        recall_per_class = matrix.diagonal()/matrix.sum(axis=1)
        precision_per_class = matrix.diagonal()/matrix.sum(axis=0)
        auc = metrics.roc_auc_score(_labels[dataset.test_indices].cpu().numpy(), nn.Softmax(dim=1)(logprobs[dataset.test_indices]).cpu().numpy(), multi_class='ovr',average='weighted')

    return auc, mic, mac, recall_per_class, precision_per_class


def test_source(model, dataset, args, insts, idx, labels, adj, device, index=-1):
    model.eval()
    _insts = insts.clone().detach().to(device)
    _labels = labels.clone().detach().cpu()
    with torch.no_grad():
        logprobs, vec, linear = model.inference(_insts, adj, idx, index)
    preds_labels = torch.max(logprobs[dataset.s_test_indices], 1)[1].squeeze_().cpu()
    pred_acc = torch.sum(preds_labels == _labels[dataset.s_test_indices]).item() / float(len(dataset.s_test_indices))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mic,mac=f1_score(_labels[dataset.s_test_indices], preds_labels, average="micro"), f1_score(_labels[dataset.s_test_indices], preds_labels, average="macro")
        matrix = confusion_matrix(_labels[dataset.s_test_indices], preds_labels)
        recall_per_class = matrix.diagonal()/matrix.sum(axis=1)
        precision_per_class = matrix.diagonal()/matrix.sum(axis=0)
        auc = metrics.roc_auc_score(_labels[dataset.s_test_indices].cpu().numpy(), nn.Softmax(dim=1)(logprobs[dataset.s_test_indices]).cpu().numpy(), multi_class='ovr',average='weighted')

    return auc, mic, mac, recall_per_class, precision_per_class

def fine_tune(model, dataset, args, target_insts, target_labels, target_adj, target_idx, device, graph, few_shot_labels):
    dataset.change_task('finetune')
    model.finetune()
    finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.finetune_lr, weight_decay=args.weight_decay)
    t_insts = target_insts.clone().detach().to(device)
    if args.batch_size == -1:
        bt_size = len(dataset)
    else:
        bt_size = args.batch_size
    f_data_loader = DataLoader(dataset, batch_size=bt_size,
                                shuffle=False, num_workers=0, collate_fn=dataset.collate)

    pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_idx, target_labels, target_adj, device, -1)
    log.add_metric(graph.name + '_pred_auc', pred_acc, 0)
    log.add_metric(graph.name + '_micro_F', mic, 0)
    log.add_metric(graph.name + '_macro_F', mac, 0)
    log.add_metric(graph.name + '_recall_per_class', recall_per_class, 0)
    log.add_metric(graph.name + '_precision_per_class', precision_per_class, 0)

    best_target_acc = 0.0
    best_epoch = 0.0
    for epoch in range(args.finetune_epoch):
        model.train()
        if epoch == args.pre_finetune:
            model.finetune_inv()
            finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.finetune_lr, weight_decay=args.weight_decay)
        running_loss = 0
        dataset.change_task('finetune')
        for batch in f_data_loader:
            finetune_optimizer.zero_grad()
            tindex = batch.to(device)
            logprobs, tvec, tlinear = model.inference(t_insts, target_adj, target_idx, index=-1)
            loss = F.nll_loss(logprobs[tindex], target_labels[tindex])
            running_loss += loss.item()
            loss.backward()
            finetune_optimizer.step()

        pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_idx, target_labels, target_adj, device, -1)
        logger.info("Iteration {}, loss = {}, auc = {}".format(epoch, running_loss/len(f_data_loader), pred_acc))
        log.add_metric(graph.name+'_finetuine_loss', running_loss/len(f_data_loader), epoch)
        log.add_metric(graph.name + '_pred_auc', pred_acc, epoch+1)
        log.add_metric(graph.name + '_micro_F', mic, epoch+1)
        log.add_metric(graph.name + '_macro_F', mac, epoch+1)
        log.add_metric(graph.name + '_recall_per_class', recall_per_class, epoch+1)
        log.add_metric(graph.name + '_precision_per_class', precision_per_class, epoch+1)
        if pred_acc > best_target_acc:
            best_target_acc = pred_acc
            best_epoch = epoch
            viz_tvec = tvec

    if args.viz == True:
        viz_single(viz_tvec, log._logdir, graph.name, best_epoch-1, best_target_acc, target_labels, few_shot_labels)
    print("=============================================================")
    line = "{} - Epoch: {}, best_target_acc: {}"\
        .format(graph.name, best_epoch, best_target_acc)
    print(line)

    return best_target_acc


def agnn_fine_tune(model, dataset, args, target_insts, target_labels, target_adj, i, device, graph, few_shot_labels):
    dataset.change_task('finetune')
    finetune_optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    t_insts = target_insts.clone().detach().to(device)
    if args.batch_size == -1:
        bt_size = len(dataset)
    else:
        bt_size = args.batch_size
    f_data_loader = DataLoader(dataset, batch_size=bt_size,
                                shuffle=True, num_workers=0, collate_fn=dataset.collate)

    pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_labels, target_adj, device, -1)
    best_target_acc = 0.0
    best_epoch = 0.0
    for epoch in tqdm(range(1000)):
        model.train()
        running_loss = 0
        dataset.change_task('finetune')
        for batch in f_data_loader:
            finetune_optimizer.zero_grad()
            tindex = batch.to(device)
            logprobs, tvec = model(t_insts, target_adj, index=-1)
            loss = F.nll_loss(logprobs[tindex], target_labels[tindex])
            running_loss += loss.item()
            loss.backward()
            finetune_optimizer.step()

        pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_labels, target_adj, device, -1)
    print("=============================================================")
    line = "{} - Epoch: {}, acc: {}"\
        .format(graph.name, epoch, pred_acc)
    print(line)

    return pred_acc

time_start = time.time()

graphs = []
input_dim, num_time = [], []
_alpha = []
datasets = args.datasets.split("+")
for i, data in enumerate(datasets):
    if args._alpha is None:
        _alpha.append(0)
    else:
        _alpha.append(float(args._alpha[i]))
    g = GraphLoader(data, sparse=True, args=args)
    graphs.append(g)
    num_time.append(g.X.shape[1])
    input_dim.append(g.X.shape[2])

dataset_ = []
data_name = []
for g in graphs:
    adj = g.adj
    actual_adj = adj.copy()
    data = MyData(actual_adj, g.Y)
    dataset_.append(MyDataset(data, "train", ratio=args.ratio, few_shot=args.few_shot))
    data_name.append(g.name)

time_end = time.time()
logger.info("Time used to process the data set = {} seconds.".format(time_end - time_start))

num_data_sets = len(dataset_)
pred_accs = []

for i in range(num_data_sets):
    graphs_ = []
    for j in range(len(graphs)):
        graphs_.append(copy.deepcopy(graphs[j]))
    dataset = []
    for j in range(len(dataset_)):
        dataset.append(copy.deepcopy(dataset_[j]))
    nclass, dataloader, ndim, num_nodes, times = [], [], [], [], []
    for j in range(num_data_sets):
        if j != i:
            nclass.append(graphs_[j].Y.cpu().numpy().max()+1)
            ndim.append(int(graphs_[j].X.shape[-1]))
            num_nodes.append(graphs_[j].X.shape[0])
            times.append(dataset[j].time)
        dataset[j].change_task('train')
        dataloader.append(DataLoader(dataset[j], batch_size=len(dataset[j]),
                                        shuffle=True, num_workers=0, collate_fn=dataset[j].collate, drop_last=True))
    nclass.append(graphs_[i].Y.cpu().numpy().max()+1)
    ndim.append(int(graphs_[i].X.shape[-1]))
    num_nodes.append(graphs_[i].X.shape[0])
    times.append(dataset[i].time)

    # Build source instances.
    source_insts = []
    source_adj = []
    source_labels = []
    source_names = []
    source_agent = []
    source_idx = []

    for j in range(num_data_sets):
        if j != i:
            # source_insts.append(graphs_[j].X[:, :-1, :])
            source_insts.append(graphs_[j].X)
            source_adj.append(graphs_[j].adj)
            source_labels.append(graphs_[j].Y)
            source_names.append(graphs_[j].name)
            source_idx.append(graphs_[j].idx)

    # Build target instances.
    target_insts = graphs_[i].X
    target_adj = graphs_[i].adj
    target_labels = graphs_[i].Y
    target_idx = graphs_[i].idx

    # pdb.set_trace()

    # set parameters for mdanet
    configs = {"input_dim": ndim, "hidden_layers": args.hidden, "num_classes": nclass, "heads":args.heads, "times":times,
                "num_epochs": args.epoch, "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "num_sources": num_data_sets - 1, 
                "ndim": args.dim, 'node_num': num_nodes, "num_class": nclass, "time": num_time[0], "layers":args.layers, "m_dim":args.m_dim,
                "dropout": args.dropout, "finetune epoch": args.finetune_epoch, "train ratio": args.ratio, "type": args.gnn, "feat_num": args.feat_num}
    num_epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    num_sources = configs["num_sources"]
    lr = configs["lr"]
    mu = configs["mu"]
    logger.info("Target domain is {}.".format(graphs_[i].name))
    logger.info("Hyperparameter setting = {}.".format(configs))

    # For visualization
    few_shot_labels = target_labels.clone()*0-1
    few_shot_labels[dataset[i].finetune_indices] = target_labels[dataset[i].finetune_indices]

    transnet = TransNet(configs).to(device)

    optimizer = optim.Adam(transnet.parameters(), lr=lr, betas=(0.5, 0.999))

    if args.weight is None:
        ass_labels = None
        best_target_acc = 0.0
        best_epoch = 0.0
        time_start = time.time()
        for t in range(num_epochs):
            p = float(1 + t) / num_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            sudo_weight = _alpha[i]
            
            train_loss, dt_loss, domain_loss, svec, tvec, sh_linear, th_linear = train_epoch(transnet, dataset, dataloader, optimizer, num_data_sets,
                                                                                            args, source_insts, target_insts, source_adj, source_labels, source_idx, 
                                                                                            target_labels, target_adj, target_idx,
                                                                                            i, device, alpha, sudo_weight, nclass, t)
            # Test on source domains
            source_acc = {}
            jj = 0
            for j in range(num_data_sets):
                if j != i:
                    pred_acc, mic, mac, recall_per_class, precision_per_class = test_source(transnet, dataset[j], args, source_insts[jj], source_idx[jj],
                        source_labels[jj], source_adj[jj], device, index=jj)
                    source_acc[source_names[jj]] = pred_acc
                    log.add_metric(graphs_[i].name+'_'+source_names[jj]+'_prediction_accuracy', pred_acc, t)
                    jj = jj + 1

            if pred_acc > best_target_acc:
                best_target_acc = pred_acc
                best_epoch = t

            logger.info("Epoch {}, train loss = {}, source_auc = {}".format(t, train_loss, source_acc))

            if t>0 and t % args.plt_i == 0 and args.viz == True:
                with torch.no_grad():
                    logprobs, _, _ = transnet.inference(target_insts, target_adj, target_idx, index=-1)
                preds_labels = torch.max(logprobs, 1)[1].squeeze_().detach().cpu()
                viz_single(tvec, log._logdir, graphs_[i].name, t, 0.0, target_labels, few_shot_labels, adapt=1, sudo_label=preds_labels, ass_node=ass_labels)

            # Save weight
            if t == num_epochs - 1 and args.save is True:
                checkpoint = {"model_state_dict": transnet.state_dict(),
                                "optimizer_state_dic": optimizer.state_dict(),
                                "loss": train_loss,
                                "epoch": t}
                path_checkpoint = "checkpoint_{}".format(t)
                if not os.path.exists(os.path.join(log._logdir, graphs_[i].name)):
                    os.makedirs(os.path.join(log._logdir, graphs_[i].name))
                torch.save(checkpoint, os.path.join(log._logdir, graphs_[i].name, path_checkpoint))
        if num_epochs > 0 and args.viz == True:
            with torch.no_grad():
                logprobs, _, _ = transnet.inference(target_insts, target_adj, target_idx, index=-1)
            preds_labels = torch.max(logprobs, 1)[1].squeeze_().detach().cpu()
            viz_single(tvec, log._logdir, graphs_[i].name, t+1, 0.0, target_labels, few_shot_labels, adapt=1, sudo_label=preds_labels, ass_node=ass_labels)
        print("=============================================================")
        line = "{} - Epoch: {}, best_target_auc: {}"\
            .format(graphs_[i].name, best_epoch, best_target_acc)
        print(line)
        time_end = time.time()
    else:
        print('Recovering from %s ...' % (args.weight[i]))
        checkpoint = torch.load(args.weight[i])
        epoch = checkpoint["epoch"]

        transnet.load_state_dict(checkpoint["model_state_dict"])

        few_shot_labels = target_labels.clone()*0-1
        few_shot_labels[dataset[i].finetune_indices] = target_labels[dataset[i].finetune_indices]

    # Finetune on target domain
    pred_acc = fine_tune(transnet, dataset[i], args, target_insts, target_labels, target_adj, target_idx, device, graphs_[i], few_shot_labels)
    pred_accs.append(pred_acc)

    logger.info("label prediction accuracy on {} = {}, time used = {} seconds.".
                format(data_name[i], pred_acc, time_end - time_start))

    dataset[i].change_task('train')
    if args.only == True:
        break
logger.info("*" * 100)

