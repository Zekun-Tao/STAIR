import warnings
import argparse
import logging
import os
import torch
import torch_sparse
import scipy.sparse as ssp
from tabulate import tabulate

from torch_geometric.data import Data
from models.base_model import MLPPredictor
from preprocess.VPs_split_Processes import construct_graph, construct_node_feature, duplicated_col, construct_rel
from models.link_prediction.STAIR import HlModual
from models.link_prediction.train import train, test


os.chdir('../..')


def echo(result, logger, epoch=0, loss=0, best=False):
    content = ('Epoch: {:<3d}, Loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
               'F1: {:.4f}'.format(epoch, loss, result['auc'], result['accuracy'],
                                   result['precision'], result['recall'], result['f1']))
    if best:
        content = ('--------   best performance    -------\n AUC: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, '
                   'Recall: {:.4f}, F1: {:.4f}'.format(result['auc'], result['accuracy'],
                                                       result['precision'], result['recall'], result['f1']))
    print(content)
    logger.info(content)


def parse_arguments():
    warnings.simplefilter("ignore", UserWarning)
    parser = argparse.ArgumentParser(description='hyper-parameter for naive-gnn')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--input_channels', type=int, default=40)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=2 * 1024)
    parser.add_argument('--gnn_batch_size', type=int, default=64 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--f_edge_dim', type=int, default=16)
    parser.add_argument('--f_node_dim', type=int, default=64)
    parser.add_argument('--g_phi_dim', type=int, default=64)

    parser.add_argument('--model_name', type=str, default='STAIR')
    parser.add_argument('--alpha', type=float, default=-1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--year', choices = [2019, 2024] , type=int, default=2019)
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args)

    folder_path = f"datasets/as_connections_{args.year}"

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'results/logs/link_prediction_{args.year}.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    logger.info(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    table_header, table_data = ['auc', 'accuracy', 'precision', 'recall', 'f1'], []
    for run in range(args.runs):
        best_performance = {'auc': .0, 'accuracy': .0, 'precision': .0, 'recall': .0, 'f1': .0}
        logger.info('############################' + str(run) + '#################################')

        train_edge_index_allpath, test_edge_index_allpath, num_node = construct_graph(folder_path)
        test_edge_index_rmDup = duplicated_col(test_edge_index_allpath, train_edge_index_allpath)

        edge_index = torch.cat((train_edge_index_allpath, test_edge_index_rmDup), dim=1)
        train_edge_index, train_edge_rel = construct_rel(train_edge_index_allpath, folder_path)
        test_edge_index_rmDup, test_edge_index_rmDup_rel = construct_rel(test_edge_index_rmDup, folder_path)

        data = Data(edge_index=train_edge_index_allpath, num_nodes=num_node)
        data.adj_t = torch_sparse.SparseTensor.from_edge_index(data.edge_index)
        data.adj_t = data.adj_t.to(device)
        data.x = construct_node_feature(folder_path)
        data.x = data.x.to(device)

        edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
        A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])),
                           shape=(data.num_nodes, data.num_nodes))

        model = HlModual(args.input_channels, args.hidden_channels,
                         args.hidden_channels, args.num_layers,
                         args.dropout, args=args).to(device)

        predictor = MLPPredictor(args.hidden_channels, args.hidden_channels, 1,
                                 args.mlp_num_layers, args.dropout).to(device)

        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data.x, data.edge_index, optimizer, args.batch_size, A, data, args)
            results = test(model, predictor, data.x, test_edge_index_rmDup, args.test_batch_size, A, data)

            if epoch % args.eval_steps == 0 or results['accuracy'] > best_performance['accuracy']:
                echo(results, logger, epoch, loss)

            if results['accuracy'] > best_performance['accuracy']:  # 记录每个run的最好成绩，保存预测结果
                best_performance = {k: results[k] for k in best_performance.keys()}

                torch.save({'edge_index': edge_index, 'num_node': num_node,
                            'train_edge': train_edge_index, 'train_rel': train_edge_rel,
                            'test_edge': test_edge_index_rmDup, 'test_rel': test_edge_index_rmDup_rel,
                            'pred_edge_index': results['edge'], 'true_label': results['true_labels'],
                            'pred_label': results['pred_labels'], 'pred_logit': results['pred_logits']},
                           f'results/outcomes/stair_link_prediction_{str(run)}_{args.year}.pth')

        echo(best_performance, logger, best=True)
        table_data.append([best_performance['auc'], best_performance['accuracy'], best_performance['precision'],
                           best_performance['recall'], best_performance['f1']])

    table = tabulate(table_data, headers=table_header, tablefmt='simple')
    print(table)
    logging.info(table)


if __name__ == "__main__":
    main()
