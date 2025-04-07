import torch
import argparse
import logging
import torch_sparse
import numpy as np
import random
from models.base_model import MLPPredictor, MLPPredictor2, GCN, GraphSAGE, GAT, Dual_GraphSAGE, Dual_GCN
from torch_geometric.data import Data
from tabulate import tabulate
from preprocess.VPs_split_Processes import construct_graph, construct_node_feature, duplicated_col, construct_rel, construct_edge_feature
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_curve
from torch.utils.data import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def echo(result, logger, epoch=0, loss=0, best=False):
    content = ('Epoch: {:<3d}, Loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
               'F1: {:.4f}, Pos_F1:{:.4f}'.format(epoch, loss, result['auc'], result['accuracy'],
                                                         result['precision'], result['recall'],
                                                    result['f1'], result['pos_label_f1']))
    if best:
        content = ('--------   best performance    -------\n'
                   'AUC: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                   'F1: {:.4f}, Pos_F1:{:.4f}'.format(result['auc'], result['accuracy'],
                                                         result['precision'], result['recall'],
                                                    result['f1'], result['pos_label_f1']))
    print(content)
    logger.info(content)


def train(model, predictor, criterion, emb, adj_t, train_edge, train_label, optimizer, batch_size):
    model.train()
    predictor.train()

    train_edge = train_edge.to(emb.device)
    train_label = train_label.to(emb.device)
    total_loss = num_examples = 0

    for perm in DataLoader(range(train_edge.size(1)), batch_size, shuffle=True):
        optimizer.zero_grad()
        edge_index = train_edge[:, perm].to(emb.device)
        label = train_label[perm].float().to(emb.device)

        out, out2, out3 = model(emb, adj_t)
        logits1 = predictor(out[edge_index[0]], out[edge_index[1]]).squeeze()
        logits2 = predictor(out2[edge_index[0]], out2[edge_index[1]]).squeeze()
        logits3 = predictor(out3[edge_index[0]], out3[edge_index[1]]).squeeze()

        loss1 = criterion(logits1, label)
        loss2 = criterion(logits2, label)
        loss3 = criterion(logits3, label)
        loss = loss1 + loss2 + loss3

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * batch_size
        num_examples += batch_size

    return total_loss / num_examples


@torch.no_grad()
def test(model, predictor, emb, adj_t, test_edge, test_label, batch_size):
    model.eval()
    predictor.eval()

    test_edge = test_edge.to(emb.device)
    test_label = test_label.to(emb.device)

    pred, labels, pred_edges = [], [], []
    for perm in DataLoader(range(test_edge.size(1)), batch_size, shuffle=True):
        edge_index = test_edge[:, perm]
        label = test_label[perm]

        out, _, _ = model(emb, adj_t)
        logits = predictor(out[edge_index[0]], out[edge_index[1]])
        pred.append(logits)
        labels.append(label)
        pred_edges.append(edge_index)

    logits = torch.cat(pred, dim=0).cpu()
    pred_edge = torch.cat(pred_edges, dim=1)
    labels = torch.cat(labels).cpu()

    auc = roc_auc_score(labels, logits)
    optimal_threshold = 0.95
    y_pred = torch.where(logits >= optimal_threshold, 1, 0)
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, pos_label=0)
    pos_label_f1 = f1_score(labels, y_pred)

    return {'auc': auc * 100, 'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100,
            'pos_label_f1': np.max(pos_label_f1) * 100, 'f1': f1 * 100, 'edge': pred_edge, 'true_labels': labels,
            'pred_labels':y_pred, 'pred_logits': pred}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Base-Graph Neural Network')
    parser.add_argument('--input_dim', type=int, default=40, help='Input layer dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of pred classes')
    parser.add_argument('--gnn_name', choices=['GCN', 'GAT', 'GraphSage', 'Dual_GraphSage', 'Dual_GCN'],
                        default='Dual_GraphSage', help="Model selection")
    parser.add_argument('--lr', default=0.0005, help="Learning Rate selection")
    parser.add_argument('--wd', default=5e-4, help="weight_decay selection")
    parser.add_argument('--run', default=20, help="num of runs")
    parser.add_argument('--epochs', default=30, help="train epochs selection")
    parser.add_argument('--eval_step', default=5, help="steps print experiment_result_stores")
    parser.add_argument('--num_node', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512 * 1024)
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args)

    folder_path = "/2019_allpath/preprocessed_path_to_connection"
    suffix = '_2019' if '2019' in folder_path else '_2024'

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'/data/taozekun/workspace/ed_rel/models/experiment_result_logs/0118_result_logs/'
                                 f'experiment_rel_epoch20_{args.gnn_name}{suffix}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logger.info(args)
    table_header, table_data = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Pos_f1'], []

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for run in range(args.run):
        best_performance = {'auc': .0, 'accuracy': .0, 'precision': .0, 'recall': .0, 'pos_label_f1': .0, 'f1': .0}
        print('#################################          ', run, '          #################################')
        logger.info('############################' + str(run) + '#################################')

        data_dict = torch.load(f'/best_performance_neo_link_{str(run)}{suffix}.pth')
        edge_index, train_edge_index, train_edge_rel = data_dict['edge_index'], data_dict['train_edge'], data_dict[
            'train_rel']
        test_edge_index_rmDup, test_edge_index_rmDup_rel = data_dict['test_edge'], data_dict['test_rel']
        num_node = data_dict['num_node']
        edge_index = torch.cat((train_edge_index, test_edge_index_rmDup), dim=1)

        data = Data(edge_index=edge_index, num_nodes=num_node)
        data.adj_t = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], sparse_sizes=(num_node, num_node))
        data.adj_t = data.adj_t.to(device)
        node_features = construct_node_feature(folder_path)
        data.x = node_features
        data.x = data.x.to(device)

        if args.gnn_name == 'GCN':
            model = GCN(args.input_dim, args.hidden_dim, args.hidden_dim, args.num_layers, args.dropout_rate).to(device)
        elif args.gnn_name == 'GAT':
            model = GAT(args.input_dim, args.hidden_dim, args.hidden_dim, args.num_layers, args.dropout_rate).to(device)
        elif args.gnn_name == 'GraphSage':
            model = GraphSAGE(args.input_dim, args.hidden_dim, args.hidden_dim, args.num_layers, args.dropout_rate).to(
                device)
        elif args.gnn_name == 'Dual_GraphSage':
            model = Dual_GraphSAGE(args.input_dim, args.hidden_dim, args.hidden_dim, args.num_layers, args.dropout_rate).to(
                device)
        elif args.gnn_name == 'Dual_GCN':
            model = Dual_GCN(args.input_dim, args.hidden_dim, args.hidden_dim, args.num_layers, args.dropout_rate).to(
                device)
        else:
            print('No such GNN model')

        predictor = MLPPredictor2(args.hidden_dim, args.hidden_dim, args.num_classes, num_layers=2,
                                 dropout=args.dropout_rate).to(device)
        criterion = torch.nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, predictor, criterion, data.x, data.adj_t, train_edge_index, train_edge_rel, optimizer,
                         args.batch_size)
            result = test(model, predictor, data.x, data.adj_t, test_edge_index_rmDup, test_edge_index_rmDup_rel,
                          args.batch_size)

            if epoch % args.eval_step == 0 or result['accuracy'] > best_performance['accuracy']:
                echo(result, logger, epoch, loss)

            if result['accuracy'] > best_performance['accuracy']:
                best_performance = {k: result[k] for k in best_performance.keys()}
                torch.save(model.state_dict(), f'datasets/pretrain_model/{args.gnn_name}_model_{run}{suffix}.pt')
                torch.save(predictor.state_dict(), f'datasets/pretrain_model/{args.gnn_name}_predictor_{run}{suffix}.pt')

        echo(best_performance, logger, best=True)

        table_data.append([best_performance['auc'], best_performance['accuracy'], best_performance['precision'],
                           best_performance['recall'], best_performance['f1'], best_performance['pos_label_f1']])

    table = tabulate(table_data, headers=table_header, tablefmt='simple')
    print(table)
    logging.info(table)


if __name__ == '__main__':
    main()
