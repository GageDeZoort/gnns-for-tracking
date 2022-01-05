import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from models.interaction_network import InteractionNetwork as IN

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

class TCN(nn.Module):
    def __init__(self, node_indim, edge_indim, xc_outdim):
        super(TCN, self).__init__()
        self.in_w1 = IN(node_indim, edge_indim, 
                        node_outdim=3, edge_outdim=4, 
                        hidden_size=80)
        self.in_w2 = IN(3, 4, 
                        node_outdim=3, edge_outdim=4, 
                        hidden_size=80)
        self.in_c1 = IN(3, 13, 
                        node_outdim=3, edge_outdim=8, 
                        hidden_size=50)
        self.in_c2 = IN(3, 8, 
                        node_outdim=3, edge_outdim=8, 
                        hidden_size=50)
        self.in_c3 = IN(3, 8, 
                        node_outdim=3, edge_outdim=8, 
                        hidden_size=50)
        self.W = MLP(12, 1, 80)
        self.B = MLP(12, 1, 80)
        self.X = MLP(12, xc_outdim, 80)
        
    def forward(self, x: Tensor, edge_index: Tensor, 
                edge_attr: Tensor) -> Tensor:
        
        # re-embed the graph twice with add aggregation
        x1, edge_attr_1 = self.in_w1(x, edge_index, edge_attr)
        x2, edge_attr_2 = self.in_w2(x1, edge_index, edge_attr_1)
        
        # combine all edge features, use to predict edge weights
        initial_edge_attr = torch.cat([edge_attr, edge_attr_1, 
                                       edge_attr_2], dim=1)
        edge_weights = torch.sigmoid(self.W(initial_edge_attr))

        # combine edge weights with original edge features
        edge_attr_w = torch.cat([edge_weights, 
                                 initial_edge_attr], dim=1)

        xc1, edge_attr_c1 = self.in_c1(x, edge_index, 
                                       edge_attr_w)
        xc2, edge_attr_c2 = self.in_c2(xc1, edge_index, 
                                       edge_attr_c1)
        xc3, edge_attr_c3 = self.in_c3(xc2, edge_index, 
                                       edge_attr_c2)
        all_xc = torch.cat([x,xc1,xc2,xc3], dim=1)
        beta = torch.sigmoid(self.B(all_xc))
        xc = self.X(all_xc)
        return edge_weights, xc, beta
        
