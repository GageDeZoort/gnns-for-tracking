import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data, Dataset

class GraphDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None,
                 graph_files=[]):
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_files = graph_files
    
    @property
    def raw_file_names(self):
        return self.graph_files

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.graph_files)
        
    def get(self, idx):
        with np.load(self.graph_files[idx]) as f:
            # for k in f.iterkeys(): print(k)
            x = torch.from_numpy(f['x'])
            edge_attr = torch.from_numpy(f['edge_attr'])
            edge_index = torch.from_numpy(f['edge_index'])
            y = torch.from_numpy(f['y'])
            particle_id = torch.from_numpy(f['particle_id'])
            if len(x)==0:
                x = torch.tensor([], dtype=torch.float)
                particle_id = torch.tensor([], dtype=torch.long)
                edge_index = torch.tensor([[],[]], dtype=torch.long)
                edge_attr = torch.tensor([], dtype=torch.float)
                y = torch.tensor([], dtype=torch.float)

            # make graph undirected
            row_0, col_0 = edge_index
            row = torch.cat([row_0, col_0], dim=0)
            col = torch.cat([col_0, row_0], dim=0)
            edge_index = torch.stack([row, col], dim=0)
            negate = torch.tensor([[-1], [-1], [-1], [1]])
            edge_attr = torch.cat([edge_attr, negate*edge_attr], dim=1)
            y = torch.cat([y,y])

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=torch.transpose(edge_attr, 0, 1),
                        y=y, particle_id=particle_id)
            data.num_nodes = len(x)

        return (data, self.graph_files[idx])  
