# system
import os
import sys
import argparse
import logging
import multiprocessing as mp
from time import time
from functools import partial
from collections import Counter

# externals
import yaml
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import optim
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

# custom modules
from models.graph_dataset import GraphDataset
from models.track_condensation_network import TCN
from models.condensation_loss import condensation_loss
from models.condensation_loss import background_loss

def parse_args():
    """ parse command line arguments
    """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', '--indir', type=str, default='graphs/train1_ptmin0')
    add_arg('-o', '--outdir', type=str, default='trained_models')
    add_arg('--outfile', type=str, default='')
    add_arg('-v', '--verbose', type=bool, default=0)
    add_arg('--n-train', type=int, default=120)
    add_arg('--n-test', type=int, default=30)
    add_arg('--n-val', type=int, default=10)
    add_arg('--learning-rate', type=float, default=10**-4)
    add_arg('--gamma', type=float, default=0.95)
    add_arg('--step-size', type=int, default=10)
    add_arg('--n-epochs', type=int, default=250)
    add_arg('--log-interval', type=int, default=10)
    add_arg('--q-min', type=float, default=0.05)
    add_arg('--sb', type=float, default=1)
    add_arg('--save-models', type=bool, default=False)
    return parser.parse_args()

def zero_div(a,b):
    """ divide, potentially by zero
    """
    return a/b if (b!=0) else 0

def validate(model, device, val_loader):
    """ validation routine, used to set edge weight 
        threshold (thld) where the true positive rate (TPR)
        equals the true negative rate (TNR)
    """
    model.eval()
    opt_thlds, accs = [], []
    for batch_idx, (data, f) in enumerate(val_loader):
        data = data.to(device)
        w, xc, beta = model(data.x, data.edge_index, data.edge_attr)
        y, w = data.y, w.squeeze()
        
        # BCE loss between edge weights (w) and targets (y)
        loss = F.binary_cross_entropy(w, y, reduction='mean').item()
        
        # define optimal threshold (thld) where TPR = TNR 
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.01, 0.5, 0.001):
            TP = torch.sum((y==1) & (w>thld)).item()
            TN = torch.sum((y==0) & (w<thld)).item()
            FP = torch.sum((y==0) & (w>=thld)).item()
            FN = torch.sum((y==1) & (w<=thld)).item()
            acc = (TP+TN)/(TP+TN+FP+FN)
            TPR = zero_div(TP, TP+FN) 
            TNR = zero_div(TN, TN+FP)
            delta = abs(TPR-TNR)
            
            # keep track of the thld minimizing abs(TPR-TNR)
            if (delta < diff): 
                diff, opt_thld, opt_acc = delta, thld, acc

        opt_thlds.append(opt_thld)
        accs.append(opt_acc)

    logging.info(f'Optimal edge weight threshold: {np.mean(opt_thlds):.5f}')
    return np.nanmean(opt_thlds) 

def test(model, device, test_loader, thld=0.5, 
         loss_c_scale=10, loss_b_scale=1./500):
    """ test routine, call on an unseen portion of data to 
        evaluate how well the model has generalized
    """
    model.eval()
    losses = []   # total loss
    losses_w = [] # edge weight loss
    accs = []     # edge classification accuracies
    losses_c = [] # condensation loss
    losses_b = [] # background loss
    with torch.no_grad():
        for batch_idx, (data,f) in enumerate(test_loader):
            data = data.to(device)
            w, xc, beta = model(data.x, data.edge_index, data.edge_attr)
            
            # edge classification accuracy
            TP = torch.sum((data.y==1).squeeze() & 
                           (w>thld).squeeze()).item()
            TN = torch.sum((data.y==0).squeeze() & 
                           (w<thld).squeeze()).item()
            FP = torch.sum((data.y==0).squeeze() &
                           (w>thld).squeeze()).item()
            FN = torch.sum((data.y==1).squeeze() & 
                           (w<thld).squeeze()).item()
            acc = zero_div(TP+TN, TP+TN+FP+FN)
            accs.append(acc)
            
            # edge weight loss
            loss = F.binary_cross_entropy(w.squeeze(1), data.y, 
                                          reduction='mean').item()
            losses_w.append(loss)
            
            # condensation loss
            particle_id = data.particle_id
            loss_c = condensation_loss(beta, xc, particle_id, 
                                       device=device, 
                                       q_min=args.q_min).item()
            loss_c *= loss_c_scale
            losses_c.append(loss_c)

            # background loss
            loss_b = background_loss(beta, xc, particle_id, 
                                     device='cpu', q_min=args.q_min, 
                                     sb=args.sb).item()
            loss_b *= loss_b_scale
            losses_b.append(loss_b)
            
            # total loss
            loss += loss_c + loss_b
            losses.append(loss)
    
    logging.info(f'Total Test Loss: {np.mean(losses):.5f}')
    logging.info(f'Edge Classification Loss: {np.mean(losses_w):.5f}')
    logging.info(f'Edge Classification Accuracy: {np.mean(accs):.5f}')
    logging.info(f'Condensation Test Loss: {np.mean(losses_c):.5f}')
    logging.info(f'Background Test Loss: {np.mean(losses_b):.5f}')
    return np.nanmean(losses), np.nanmean(accs)

def train(args, model, device, train_loader, optimizer, epoch,
          loss_w_scale=1, loss_c_scale=10, loss_b_scale=1./500):
    """ train routine, loss and accumulated gradients used to update
        the model via the ADAM optimizer externally 
    """
    model.train()
    epoch_t0 = time()
    losses = []   # total loss
    losses_w = [] # edge weight loss
    losses_c = [] # condensation loss
    losses_b = [] # background loss
    for batch_idx, (data, f) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        w, xc, beta = model(data.x, data.edge_index, data.edge_attr)
        y, w = data.y, w.squeeze(1)
        particle_id = data.particle_id

        # edge weight loss
        loss_w = F.binary_cross_entropy(w, y, reduction='mean')
        loss_w *= loss_w_scale
        loss = loss_w

        # condensation loss
        loss_c = condensation_loss(beta, xc, particle_id, 
                                   device=device, q_min=args.q_min)
        loss_c *= loss_c_scale
        
        # background loss
        loss_b = background_loss(beta, xc, particle_id, 
                                 device='cpu', q_min=args.q_min, 
                                 sb=args.sb)
        loss_b *= loss_b_scale
 
        # optimize total loss
        loss += (loss_c + loss_b)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                        .format(epoch, batch_idx, len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), 
                                loss.item()))
        
        # store losses
        losses.append(loss.item())
        losses_w.append(loss_w.item())
        losses_c.append(loss_c.item())
        losses_b.append(loss_b.item())

    logging.info(f"Epoch {epoch} Time: {(time()-epoch_t0):.4f}s")
    loss = np.nanmean(losses)
    loss_w = np.nanmean(losses_w)
    loss_c = np.nanmean(losses_c)
    loss_b = np.nanmean(losses_b)
    logging.info(f"Epoch {epoch} Train Loss: {loss:.6f}")
    logging.info(f"Epoch {epoch}: Edge Weight Loss: {loss_w:.6f}")
    logging.info(f"Epoch {epoch}: Condensation Loss: {loss_c:.6f}")
    logging.info(f"Epoch {epoch}: Background Loss: {loss_b:.6f}")
    return loss, loss_w, loss_c, loss_b

def write_output_file(fname, args, df):
    f = open(fname, 'w')
    f.write('# args used in training routine:\n')
    for arg in vars(args):
        f.write(f'# {arg}: {getattr(args,arg)}\n')
    df.to_csv(f)
    f.close()
    

# parse the command line 
args = parse_args()

# initialize logging
log_format = '%(asctime)s %(levelname)s %(message)s'
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(level=log_level, format=log_format)
logging.info('Initializing')

# import graphs from specified directory 
graph_files = np.array(os.listdir(args.indir))
graph_paths = np.array([os.path.join(args.indir, graph_file)
                        for graph_file in graph_files])

# create partion graphs randomly into train, test, val sets
n_graphs = len(graph_files)
IDs = np.arange(n_graphs)
np.random.shuffle(IDs)
ntrn = args.n_train
ntst = args.n_test
nval = args.n_val
train_IDs = IDs[:ntrn]
test_IDs = IDs[ntrn:(ntrn+ntst)]
val_IDs = IDs[(ntrn+ntst):(ntrn+ntst+nval)]
partition = {'train': graph_paths[train_IDs],
             'test':  graph_paths[test_IDs],
             'val': graph_paths[val_IDs]}

# build data loaders for each data partition
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
train_set = GraphDataset(graph_files=partition['train'])
train_loader = DataLoader(train_set, **params)
test_set = GraphDataset(graph_files=partition['test'])
test_loader = DataLoader(test_set, **params)
val_set = GraphDataset(graph_files=partition['val'])
val_loader = DataLoader(val_set, **params)

# use cuda (gpu) if possible, otherwise fallback to cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logging.info(f'Utilizing {device}')

# instantiate instance of the track condensation network
model = TCN(3, 4, 2).to(device)
total_trainable_params = sum(p.numel() for p in model.parameters())
logging.info(f'Total Trainable Params: {total_trainable_params}')

# instantiate optimizer with scheduled learning rate decay
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=args.step_size,
                   gamma=args.gamma)

# training loop
output = {'train_loss': [], 'test_loss': [], 'test_acc': [],
          'train_loss_w': [], 'train_loss_c': [], 'train_loss_b': []}
for epoch in range(1, args.n_epochs + 1):
    logging.info(f"---- Epoch {epoch} ----")
    train_loss, tlw, tlc, tlb = train(args, model, device, 
                                      train_loader, optimizer, epoch)
    thld = validate(model, device, val_loader)
    test_loss, test_acc = test(model, device, test_loader, thld=thld)
    scheduler.step()

    # save output
    output['train_loss'].append(train_loss)
    output['train_loss_w'].append(tlw)
    output['train_loss_c'].append(tlc)
    output['train_loss_b'].append(tlb)
    output['test_loss'].append(test_loss)
    output['test_acc'].append(test_acc)

    outfile = 'dev'
    if (args.outfile != ''):
        outfile = args.outfile
    if (args.save_models):    
        torch.save(model.state_dict(),
                   f"trained_models/{outfile}_epoch{epoch}.pt")
    write_output_file('trained_models/train_stats/{outfile}.csv', 
                      args, pd.DataFrame(output))
   
