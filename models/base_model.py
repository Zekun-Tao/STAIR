import torch
import numpy as np
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn.functional as F


def perturb_node_features(x, noise_level=0.005):
    x = x + noise_level * torch.randn_like(x)
    return x


def mask_node_features(x, mask_ratio=0.1):
    num_node, num_features = x.size()
    mask = np.random.binomial(1, mask_ratio, (num_node, num_features))
    mask = torch.FloatTensor(mask).to(x.device)
    x = x * (1-mask)
    return x


def adjust_edge_weights_by_similarity(x, row_indices, col_indices, sim_metrics, edge_attr=None):
    pass


class Attention(nn.Module):
    def __init__(self, in_channels, hidden_size = 64):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class GatedAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(GatedAttention, self).__init__()
        self.query_linear = nn.Linear(in_size, hidden_size)
        self.key_linear = nn.Linear(in_size, hidden_size)
        self.value_linear = nn.Linear(in_size, hidden_size)
        self.gate_linear = nn.Linear(in_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.final_linear = nn.Linear(hidden_size, in_size)

    def forward(self, z):
        query = self.query_linear(z)
        key = self.key_linear(z)
        value = self.value_linear(z)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        gate = self.sigmoid(self.gate_linear(z))
        gate = gate
        gated_value = value * gate
        gated_attention_output = torch.matmul(attention_weights, gated_value)
        output = self.final_linear(gated_attention_output.sum(dim=1))
        return output, attention_weights


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.attention = GatedAttention(hidden_channels)

        self.dropout = dropout

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x


class Dual_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(Dual_GCN, self).__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.attention = Attention(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):  
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        pass
        


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):  
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class Dual_GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(Dual_GraphSAGE, self).__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.attention = Attention(hidden_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        # channel 1
        pass



class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GAT, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels))

    def reset_parameters(self):  
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class MLPPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super(MLPPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.dropout = dropout

        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, num_classes))

    def reset_parameters(self):  # 4. Reset the parameters
        for conv in self.lins:
            conv.reset_parameters()

    def forward(self, x_i, x_j=None):
        if x_j is None:
            x = x_i
        else:
            x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class MLPPredictor2(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super(MLPPredictor2, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.dropout = dropout

        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, num_classes))
        self.attention = GatedAttention(hidden_channels)

    def reset_parameters(self):  # 4. Reset the parameters
        for conv in self.lins:
            conv.reset_parameters()

    def forward(self, x_i, x_j=None):
        pass
