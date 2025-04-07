import torch
import numpy as np

from torch_scatter import scatter_add
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve


def train(model, predictor, x, train_edge_index, optimizer, batch_size, A, data, args):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    count = 0

    for perm, perm_large in zip(DataLoader(range(train_edge_index.size(1)), batch_size, shuffle=True),
                                DataLoader(range(train_edge_index.size(1)), args.gnn_batch_size, shuffle=True)):
        optimizer.zero_grad()
        edge = train_edge_index[:,perm].to(x.device)
        edge_large = train_edge_index[:,perm_large].to(x.device)
        pos_out, pos_out_struct, _, _ = model(edge, data, A, predictor, emb=x)
        _, _, pos_out_feat_large = model(edge_large, data, A, predictor, emb=x, only_feature=True)

        edge = negative_sampling(train_edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense').to(x.device)

        edge_large = negative_sampling(train_edge_index, num_nodes=x.size(0),
                                       num_neg_samples=perm_large.size(0), method='dense').to(x.device)

        neg_out, neg_out_struct, _, _ = model(edge, data, A, predictor, emb=x)
        _, _, neg_out_feat_large = model(edge_large, data, A, predictor, emb=x, only_feature=True)

        # print(pos_out.shape, pos_out_struct.shape, pos_out_feat_large.shape)
        pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
        loss1 = pos_loss + neg_loss
        pos_loss = -torch.log(pos_out_feat_large + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out_feat_large + 1e-15).mean()
        loss2 = pos_loss + neg_loss
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss3 = pos_loss + neg_loss
        loss = loss1 + loss2 + loss3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        count += 1

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, test_edge_index, batch_size, A, data):
    model.eval()
    predictor.eval()

    h = model.forward_feature(x, data.adj_t)

    edge_weight = torch.from_numpy(A.data).to(h.device)
    edge_weight = model.f_edge(edge_weight.unsqueeze(-1))

    row, col = A.nonzero()
    edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(h.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=data.adj_t.size(0))
    deg = model.f_node(deg).squeeze()

    deg = deg.cpu().numpy()
    A_ = A.multiply(deg).tocsr()

    alpha = torch.softmax(model.alpha, dim=0).cpu()
    print(alpha)

    all_pos_edge = torch.cat([edge_index.cpu(), test_edge_index], dim=1)
    test_edge_index_neg = negative_sampling(all_pos_edge, num_nodes=x.size(0),
                                            num_neg_samples=test_edge_index.size(1), method='dense')

    pos_test_preds, pos_edges = [], []
    for perm in DataLoader(range(test_edge_index.size(1)), batch_size):
        edge = test_edge_index[:,perm]
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

        src, dst = test_edge_index[:,perm].cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())
        cur_scores = alpha[0] * cur_scores + alpha[1] * gnn_scores
        pos_test_preds += [cur_scores]
        pos_edges.append(edge)

    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    pos_edge = torch.cat(pos_edges, dim=1)

    neg_test_preds, neg_edges = [], []
    for perm in DataLoader(range(test_edge_index_neg.size(1)), batch_size):
        edge = test_edge_index_neg[:,perm]
        gnn_scores = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

        src, dst = test_edge_index_neg[:,perm].cpu()
        cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
        cur_scores = torch.sigmoid(model.g_phi(cur_scores).squeeze().cpu())
        cur_scores = alpha[0] * cur_scores + alpha[1] * gnn_scores
        neg_test_preds += [cur_scores]
        neg_edges.append(edge)

    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    neg_edge = torch.cat(neg_edges, dim=1)

    labels = torch.cat((torch.ones(pos_test_pred.shape[0]), torch.zeros(neg_test_pred.shape[0])), dim=0)
    pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)
    pred_edges_index = torch.cat((pos_edge, neg_edge), dim=1)
    auc = roc_auc_score(labels, pred)

    precision, recall, thresholds = precision_recall_curve(labels, pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1)]
    y_pred = torch.where(pred >= optimal_threshold, 1, 0)
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)

    del edge_weight
    torch.cuda.empty_cache()
    return {'auc': auc * 100, 'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100,
            'f1': np.max(f1) * 100, 'edge': pred_edges_index, 'true_labels': labels, 'pred_labels':y_pred, 'pred_logits':pred}

