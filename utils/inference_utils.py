import os
import sys

sys.path.append("../")
from os.path import join

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from models import track_condensation_network as TCN
from models import interaction_network as IN


def load_pretrained_IN(
    indir,
    node_indim,
    edge_indim,
    node_outdim=3,
    edge_outdim=4,
    device="cpu",
    hidden_size=80,
):
    gnn = IN(
        node_indim,
        edge_indim,
        node_outdim=node_outdim,
        edge_outdim=edge_outdim,
        hidden_size=hidden_size,
    ).to(device)
    gnn.load_state_dict(torch.load(indir))
    return gnn


def load_pretrained_TCN(indir, node_indim, edge_indim, node_outdim, device="cpu"):
    gnn = TCN(node_indim, edge_indim, node_outdim)
    gnn.load_state_dict(torch.load(indir))
    return gnn


def zero_divide(a, b):
    if b == 0:
        return 0
    return a / b


def binary_classification_stats(output, y, thld):
    TP = torch.sum((y == 1) & (output > thld))
    TN = torch.sum((y == 0) & (output < thld))
    FP = torch.sum((y == 0) & (output > thld))
    FN = torch.sum((y == 1) & (output < thld))
    acc = zero_divide(TP + TN, TP + TN + FP + FN)
    TPR = zero_divide(TP, TP + FN)
    TNR = zero_divide(TN, TN + FP)
    return acc, TPR, TNR
