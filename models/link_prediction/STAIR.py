import torch
from torch_geometric.nn import GCNConv, SAGEConv
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
import torch.nn.functional as F
from models.base_model import perturb_node_features, mask_node_features


class HlModual(torch.nn.Module):
    pass


