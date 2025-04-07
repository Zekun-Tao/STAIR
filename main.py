import torch
import argparse
import logging
import torch_sparse
import warnings
import scipy.sparse as ssp
from models.relationship_prediction.train_test import train, test
from models.base_model import MLPPredictor2, adjust_edge_weights_by_similarity
from models.relationship_prediction.STAIR_dual_architecture import RIModual
from torch_geometric.data import Data
from tabulate import tabulate
from preprocess.VPs_split_Processes import construct_graph, construct_node_feature, duplicated_col, construct_rel

warnings.filterwarnings('ignore')


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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters-Graph Neural Network')
    parser.add_argument('--input_dim', type=int, default=40, help='Input layer dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of pred classes')
    parser.add_argument('--model', choices=['GCN', 'GAT', 'SAGE', 'ChebNet', 'TransformerConv'], default='GraphSAGE',
                        help="Model selection")
    parser.add_argument('--lr', default=0.001, help="Learning Rate selection")
    parser.add_argument('--wd', default=5e-4, help="weight_decay selection")
    parser.add_argument('--run', default=20, help="num of runs")
    parser.add_argument('--epochs', default=250, help="train epochs selection")
    parser.add_argument('--eval_step', default=1, help="steps print experiment_result_stores")
    parser.add_argument('--f_edge_dim', type=int, default=8)
    parser.add_argument('--f_node_dim', type=int, default=128)
    parser.add_argument('--g_phi_dim', type=int, default=256)
    parser.add_argument('--num_node', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2 * 1024)
    parser.add_argument('--gnn_batch_size', type=int, default=64 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=64 * 1024)
    parser.add_argument('--year', type=int, choices=[2019, 2024], default=2019)
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args)

    folder_path = f"datasets/as_connections_{args.year}"

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'results/logs/relationship_inference_{args.year}.log',
                        level=logging.INFO, format='%(asctime)s - %(message)s')
    logger.info(args)

    table_header, table_data = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Pos_f1'], []
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for run in range(args.run):
        best_performance = {'auc': .0, 'accuracy': .0, 'precision': .0, 'recall': .0, 'f1': .0, 'pos_label_f1': .0}
        print('#################################          ', run, '          #################################')
        logger.info('############################' + str(run) + '#################################')

        train_edge_index_allpath, test_edge_index_allpath, num_node = construct_graph(folder_path)
        test_edge_index_rmDup = duplicated_col(test_edge_index_allpath, train_edge_index_allpath)

        edge_index = torch.cat((train_edge_index_allpath, test_edge_index_rmDup), dim=1)
        train_edge_index, train_edge_rel = construct_rel(train_edge_index_allpath, folder_path)
        test_edge_index_rmDup, test_edge_index_rmDup_rel = construct_rel(test_edge_index_rmDup, folder_path)

        data = Data(edge_index=edge_index, num_nodes=num_node)
        data.adj_t = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], sparse_sizes=(num_node, num_node))
        data.adj_t = data.adj_t.to(device)
        data.x = construct_node_feature(folder_path)
        args.num_node = num_node

        edge_weight_eud = adjust_edge_weights_by_similarity(data.x, data.edge_index[0], data.edge_index[1], 'cos')
        edge_weight_sim = adjust_edge_weights_by_similarity(data.x, data.edge_index[0], data.edge_index[1], 'eud')

        A1 = ssp.csr_matrix((edge_weight_eud + 1e-15, (data.edge_index[0], data.edge_index[1])),
                           shape=(data.num_nodes, data.num_nodes))
        A2 = ssp.csr_matrix((edge_weight_sim + 1e-15, (data.edge_index[0], data.edge_index[1])),
                           shape=(data.num_nodes, data.num_nodes))

        data.x = data.x.to(device)
        model = RIModual(args.input_dim, args.hidden_dim, args.hidden_dim, args.num_layers, args.dropout_rate,
                         args=args).to(device)
        predictor = MLPPredictor2(args.hidden_dim, args.hidden_dim, args.num_classes, num_layers=2,
                                 dropout=args.dropout_rate).to(device)

        criterion = torch.nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) +
                                     list(predictor.parameters()), lr=args.lr)
        model.reset_parameters()
        predictor.reset_parameters()

        pretrain_model = f'datasets/data_augmentation/pretrain_model/Dual_GraphSage_model_{run}{suffix}.pt'
        model.load_state_dict(torch.load(pretrain_model), strict=False)
        pretrain_predictor = f'datasets/data_augmentation/pretrain_model/Dual_GraphSage_predictor_{run}{suffix}.pt'
        predictor.load_state_dict(torch.load(pretrain_predictor))

        for epoch in range(1, args.epochs + 1):
            loss = train(model, predictor, criterion, data, A1, A2, train_edge_index, train_edge_rel, optimizer, args)
            result = test(model, predictor, data, A1, A2, test_edge_index_rmDup, test_edge_index_rmDup_rel, args.test_batch_size)

            if epoch % args.eval_step == 0 or result['accuracy'] > best_performance['accuracy']:
                echo(result, logger, epoch, loss)

            if result['accuracy'] > best_performance['accuracy']:
                best_performance = {k: result[k] for k in best_performance.keys()}
                torch.save({'edge_index': edge_index,
                            'pred_edge_index': result['edge'], 'true_label': result['true_labels'],
                            'pred_label': result['pred_labels'], 'pred_logit': result['pred_logits']},
                           f'/datasets/data_augmentation/neo_save/best_performance_neo_rel_{str(run)}{suffix}.pth')
                torch.save(model.state_dict(), f'datasets/data_augmentation/neo_save/rel_neo_model_{run}{suffix}.pt')

        echo(best_performance, logger, best=True)
        table_data.append([best_performance['auc'], best_performance['accuracy'], best_performance['precision'],
                           best_performance['recall'], best_performance['f1'], best_performance['pos_label_f1']])

    table = tabulate(table_data, headers=table_header, tablefmt='simple')
    print(table)
    logging.info(table)


if __name__ == '__main__':
    main()
