import torch
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
import torch.nn.functional as F
from models.base_model import (Attention, adjust_edge_weights_by_similarity,
                               mask_node_features, perturb_node_features)


class RIModual(torch.nn.Module):
    pass
