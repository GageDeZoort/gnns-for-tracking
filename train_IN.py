import os 
import sys
import argparse
import logging
from time import time
from os.path import join

import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import StepLR

from models.edge_classifier_1 import EdgeClassifier
from utils.data_utils import *
from utils.train_utils import *

initialize_logger(verbose=False)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses, t0, N = [], time(), len(train_loader)
    for batch_idx, (data, fname) in enumerate(train_loader):
        data = data.to(device)
        if (len(data.x)==0): continue
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze(1)
        loss = F.binary_cross_entropy(output, y, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            percent_complete = 100. * batch_idx / N
            logging.info(f'Train Epoch: {epoch} [{batch_idx}/{N}' +
                         f'({percent_complete:.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run: break
        losses.append(loss.item())
    logging.info(f'Epoch completed in {time()-t0}s')
    logging.info(f'Train Loss: {np.nanmean(losses)}')
    return np.nanmean(losses)

def validate(model, device, val_loader):
    model.eval()
    opt_thlds, accs = [], []
    for batch_idx, (data, fname) in enumerate(val_loader):
        data = data.to(device)
        if (len(data.x)==0): continue
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze()
        loss = F.binary_cross_entropy(output, y, reduction='mean').item()
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.01, 0.6, 0.01):
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            delta = abs(TPR-TNR)
            if (delta.item() < diff): 
                diff, opt_thld, opt_acc = delta.item(), thld, acc.item()
        opt_thlds.append(opt_thld)
        accs.append(opt_acc)
    logging.info(f'Validation set accuracy (where TPR=TNR): {np.nanmean(accs)}')
    logging.info(f'Validation set optimal edge weight thld: {np.nanmean(opt_thld)}')
    return np.nanmean(opt_thlds) 

def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch_idx, (data, fname) in enumerate(test_loader):
            data = data.to(device)
            if (len(data.x)==0): continue
            output = model(data.x, data.edge_index, data.edge_attr)
            y, output = data.y, output.squeeze()
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            loss = F.binary_cross_entropy(output, data.y, 
                                          reduction='mean')
            accs.append(acc.item())
            losses.append(loss.item())
    logging.info(f'Test loss: {np.nanmean(losses):.4f}')
    logging.info(f'Test accuracy: {np.nanmean(accs):.4f}')
    return np.nanmean(losses), np.nanmean(accs)

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('-i', '--indir', type=str, default='graphs/train1_ptmin1',
                        help='input graph directory')
    parser.add_argument('--n-train', type=int, default=8000)
    parser.add_argument('--n-test', type=int, default=2000)
    parser.add_argument('--n-val', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model-outdir', type=str, default='train_output/models/EC1/',
                        help='directory in which to save models')
    parser.add_argument('--stats-outdir', type=str, default='train_output/stats/EC1/',
                        help='directory in which to save stats')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=75, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 5*10**-4)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='Learning rate step size')
    parser.add_argument('--pt-min', type=str, default='1',
                        help='Cutoff pt value in GeV (default: 1 GeV)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--sample', type=int, default=1, 
                        help='TrackML train_{} sample to train on')
    parser.add_argument('--hidden-size', type=int, default=40,
                        help='Number of hidden units per layer')

    args = parser.parse_args()
    job_name = f'IN_pt{args.pt_min}'

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    logging.info(f'Parameter use_cuda={use_cuda}')
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
        
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4}
    loaders = get_dataloaders(args.indir, args.n_train, args.n_test,
                              n_val=args.n_val, shuffle=False, 
                              params=params)
        
    model = EdgeClassifier(3, 4).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Trainable params in network: {total_trainable_params}')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)

    output = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        logging.info(f'Entering epoch {epoch}')
        train_loss = train(args, model, device, loaders['train'], optimizer, epoch)
        thld = validate(model, device, loaders['val'])
        logging.info(f'Sending thld={thld} to test routine.')
        test_loss, test_acc = test(model, device, loaders['test'], thld=thld)
        scheduler.step()
        if args.save_model:
            model_name = join(args.model_outdir,
                              job_name + f'_epoch{epoch}')
            torch.save(model.state_dict(), model_name)
        output['train_loss'].append(train_loss)
        output['test_loss'].append(test_loss)
        output['test_acc'].append(test_acc)
        np.save(join(args.stats_outdir, job_name), output)


if __name__ == '__main__':
    main()



