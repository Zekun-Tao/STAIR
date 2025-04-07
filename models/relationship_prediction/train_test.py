import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_scatter import scatter_add
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve


def train(model, predictor, criterion, data, A1, A2, train_edge, train_label, optimizer, args):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    count = 0

    train_edge = train_edge.to(data.x.device)
    train_label = train_label.to(data.x.device)

    for perm, perm_large in zip(DataLoader(range(train_edge.size(1)), args.batch_size, shuffle=True),
                                DataLoader(range(train_edge.size(1)), args.gnn_batch_size, shuffle=True)):
        optimizer.zero_grad()

        train_edge_index = train_edge[:, perm].to(data.x.device)
        train_edge_index_large = train_edge[:, perm].to(data.x.device)
        label = train_label[perm].to(data.x.device).unsqueeze(1).double()
        label_large = train_label[perm].to(data.x.device).unsqueeze(1).double()

        pos_out, pos_out_struct, _, _ = model(train_edge_index, data, A1, A2, predictor, emb=data.x)
        _, _, pos_out_feat_large = model(train_edge_index_large, data, A1, A2, predictor, emb=data.x, only_feature=True)

        loss1 = criterion(pos_out, label)
        loss2 = criterion(pos_out_struct, label)
        loss3 = criterion(pos_out_feat_large.double(), label_large)

        loss = loss1+loss2+loss3
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        count += 1

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, A1, A2, test_edge, test_label, batch_size):
    model.eval()
    predictor.eval()

    h = model.forward_feature(data.x, data.adj_t)

    edge_weight1 = torch.from_numpy(A1.data).to(h.device).double()
    edge_weight1 = model.f_edge(edge_weight1.unsqueeze(-1))

    edge_weight2 = torch.from_numpy(A2.data).to(h.device).double()
    edge_weight2 = model.f_edge(edge_weight2.unsqueeze(-1))

    row, col = A1.nonzero()
    edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(h.device)
    row, col = edge_index[0], edge_index[1]
    deg1 = scatter_add(edge_weight1, col, dim=0, dim_size=data.adj_t.size(0))
    deg1 = model.f_node(deg1).squeeze()

    deg2 = scatter_add(edge_weight2, col, dim=0, dim_size=data.adj_t.size(0))
    deg2 = model.f_node(deg2).squeeze()

    deg1 = deg1.cpu().numpy()
    deg2 = deg2.cpu().numpy()
    A1_ = A1.multiply(deg1).tocsr()
    A2_ = A2.multiply(deg2).tocsr()

    alpha = torch.tensor([0.5, 0.5])

    pos_test_preds, pos_test_label, pred_edges_index = [], [], []
    for perm in DataLoader(range(test_edge.size(1)), batch_size):
        edge = test_edge[:, perm]
        label = test_label[perm]
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

        src, dst = test_edge[:, perm].cpu()
        cur_scores1 = torch.from_numpy(np.sum(A1_[src].multiply(A1_[dst]), 1)).to(h.device)
        cur_scores2 = torch.from_numpy(np.sum(A2_[src].multiply(A2_[dst]), 1)).to(h.device)
        cur_scores = torch.cat([cur_scores1, cur_scores2], dim=1)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())
        cur_scores = alpha[0] * cur_scores + alpha[1] * gnn_scores
        pos_test_preds.append(cur_scores)
        pos_test_label.append(label)
        pred_edges_index.append(edge)

    pred = torch.cat(pos_test_preds, dim=0)
    labels = torch.cat(pos_test_label).cpu()
    pred_edge = torch.cat(pred_edges_index, dim=1)
    auc = roc_auc_score(labels, pred)

    precision, recall, thresholds = precision_recall_curve(labels, pred)
    optimal_threshold = 0.95
    y_pred = torch.where(pred >= optimal_threshold, 1, 0)
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, pos_label=0)
    pos_label_f1 = f1_score(labels, y_pred)

    return {'auc': auc * 100, 'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100,
            'pos_label_f1': np.max(pos_label_f1) * 100, 'f1': f1 * 100, 'edge': pred_edge, 'true_labels': labels,
            'pred_labels':y_pred, 'pred_logits':pred}





