import os
import json
import logging

import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def read_txt_files(folder_path):
    txt_path = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                txt_path.append(file_path)
    return txt_path


def node_from_file(txt_list):
    src_node = []
    dst_node = []
    for file in tqdm(txt_list):
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                src_node.append(line[0])
                dst_node.append(line[1])

    return src_node, dst_node


def construct_all_nodes(folder_path):
    all_txt = read_txt_files(folder_path)
    src_node, dst_node = node_from_file(all_txt)
    nodes = np.array(src_node + dst_node, dtype=int)
    nodes = np.unique(nodes).tolist()
    
    node_to_idx = {}
    idx_to_node = []
    
    for idx, node in enumerate(nodes):
        idx_to_node.append(node)
        node_to_idx[node] = len(idx_to_node) - 1

    return idx_to_node, node_to_idx


def construct_edge_index(file_list, node_to_idx):
    src_node, dst_node = node_from_file(file_list)
    src_id = list(map(lambda x: node_to_idx[x], src_node))
    dst_id = list(map(lambda x: node_to_idx[x], dst_node))

    edge = np.column_stack([src_id, dst_id])
    edge_index = edge.transpose()
    edge_index = np.unique(edge_index, axis=1)

    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    edge_index = np.unique(edge_index, axis=1)

    return edge_index


def construct_graph(folder_path):
    suffix = '_2019' if '2019' in folder_path else '_2024'

    if os.path.exists(f"datasets/node_index{suffix}.json"):
        with open(f'datasets/node_index{suffix}.json', 'r') as file:
            my_dict = json.load(file)
        idx_to_node = my_dict['idx_to_node']
        node_to_idx = my_dict['node_to_idx']
    else:
        print(f"-----------constructing ：'node_index{suffix}.json'--------------")
        idx_to_node, node_to_idx = construct_all_nodes(folder_path)
        print("共有{}个ASN...".format(len(idx_to_node)))
        with open(f'node_index{suffix}.json', 'w') as file:
            json.dump({'idx_to_node': idx_to_node, 'node_to_idx': node_to_idx}, file)

    all_txt = read_txt_files(folder_path)
    train_files, test_files = train_test_split(all_txt, test_size=0.5)
    train_edge_index = construct_edge_index(train_files, node_to_idx)
    test_edge_index = construct_edge_index(test_files, node_to_idx)
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long)
    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long)

    return train_edge_index, test_edge_index, len(idx_to_node)


def construct_node_feature(folder_path):
    suffix = '_2019' if '2019' in folder_path else '_2024'

    node_index_file = f"datasets/node_index{suffix}.json"
    node_feature_pt_file = f"datasets/node_feature_tensor{suffix}.pt"
    node_feature_csv_file = f"datasets/node_features{suffix}.csv"

    if os.path.exists(node_feature_pt_file):
        return torch.load(node_feature_pt_file)

    if not os.path.exists(node_feature_pt_file) or not(os.path.exists(node_feature_csv_file) and os.path.exists(node_index_file)):
        raise FileNotFoundError(f"missing 'node_index{suffix}.json' or 'node_features{suffix}.csv' file")
    else:
        with open(node_index_file, 'r') as file:
            my_dict = json.load(file)
        idx_to_node = my_dict['idx_to_node']

    node_feature_list = []
    node_feature = pd.read_csv(node_feature_csv_file, index_col='asn_asn')
    for node in tqdm(idx_to_node):
        if node in node_feature.index:
            node_feature_list.append(torch.tensor(node_feature.loc[node].values, dtype=torch.float32))
        else:
            node_feature_list.append(torch.zeros(node_feature.shape[1], dtype=torch.float32))
    node_feature_tensor = torch.stack(node_feature_list, dim=0)
    torch.save(node_feature_tensor, node_feature_pt_file)

    return node_feature_tensor


def duplicated_col(tensor_test, tensor_train):
    test_edge, train_edge = [], []
    for i in tensor_test.T.tolist():
        test_edge.append(tuple(i))
    for i in tensor_train.T.tolist():
        train_edge.append(tuple(i))
    count = len(set(test_edge) & set(train_edge))
    test_edge_unique = set(test_edge) - set(train_edge)
    test_edge_unique = torch.tensor(list(test_edge_unique), dtype=torch.long).T

    print("num of test edges:{} | num of train edges:{}| num of repeated edges:{}".format(tensor_test.size(1),
                                                                                          tensor_train.size(1), count))
    logger.info("num of test edges:{} | num of train edges:{}| num of repeated edges:{}".format(tensor_test.size(1),
                                                                                                tensor_train.size(1),
                                                                                                count))

    return test_edge_unique


"""constructing as rel..."""
def read_rel_file(file_path, folder_path):
    suffix = '_2019' if '2019' in folder_path else '_2024'
    node_index_file = f'/data/taozekun/workspace/ed_rel/datasets/node_index{suffix}.json'

    if os.path.exists(node_index_file):
        with open(node_index_file, 'r') as file:
            my_dict = json.load(file)
        node_to_idx = my_dict['node_to_idx']
    else:
        raise FileNotFoundError(f"missing '{node_index_file}' ")

    rel_list = []
    for file in os.listdir(file_path):
        file_addr = os.path.join(file_path, file)
        with open(file_addr, 'r') as f:
            for line in tqdm(f.readlines()):
                if line.startswith('#'):
                    continue
                else:
                    line = line.strip()
                    split_line = line.split('|')[:3]
                    rel_list.append(split_line)

    df = pd.DataFrame(rel_list, columns=['src', 'dst', 'rel'])
    df_unique = df.drop_duplicates()
    df_unique[['src', 'dst']] = df_unique[['src', 'dst']].applymap(node_to_idx.get)
    df_unique.to_csv(f'/data/taozekun/workspace/ed_rel/datasets/rel_of_edge_index{suffix}.csv', index=False)

    return df_unique


def construct_rel(edge_index, folder_path):
    suffix = '_2019' if '2019' in folder_path else '_2024'
    rel_edge_index_file = f'datasets/rel_of_edge_index{suffix}.csv'

    if os.path.exists(rel_edge_index_file):
        rel_df = pd.read_csv(rel_edge_index_file)
    else:
        raise FileNotFoundError(f"missing 'datasets/rel_of_edge_index{suffix}.csv' file ")

    rel_df['rel'] = rel_df['rel'].replace(-1, 1)
    rel_dict = rel_df.set_index(['src', 'dst'])['rel'].to_dict()
    edge_index_rel = []
    edge_index_with_rel = []
    for edge in tqdm(edge_index.T.tolist()):
        if tuple(edge) in rel_dict:
            edge_index_rel.append(rel_dict[tuple(edge)])
            edge_index_with_rel.append(edge)
        else:
            continue

    non_rel_count = (edge_index.size(1) / 2 ) - len(edge_index_rel)
    edge_index_with_rel = torch.tensor(edge_index_with_rel, dtype=torch.long).T
    edge_index_rel = torch.tensor(edge_index_rel, dtype=torch.long)
    unique_values, counts = torch.unique(edge_index_rel, return_counts=True)
    for value, count in zip(unique_values, counts):
        print("Value {} appears {} times, prop {:.4f}.".format(value, count, count/len(edge_index_rel)))

    return edge_index_with_rel, edge_index_rel


def construct_edge_feature(edges, folder_path):
    suffix = '_2019' if '2019' in folder_path else '_2024'
    edge_feature_csv_file = f'/data/taozekun/workspace/ed_rel/datasets/edge_supplementary_features{suffix}.csv'

    if not os.path.exists(edge_feature_csv_file):
        raise FileNotFoundError(f"文件 'edge_supplementary_features{suffix}.csv' 不存在，无法继续执行函数。")

    edge_feature_list = []
    edge_feature = pd.read_csv(edge_feature_csv_file, index_col='edge_index')
    for e in tqdm(edges.T.tolist()):
        if tuple(e) in edge_feature.index:
            edge_feature_list.append(torch.tensor(edge_feature.loc[tuple(e)].values, dtype=torch.float32))
        else:
            edge_feature_list.append(torch.zeros(edge_feature.shape[1], dtype=torch.float32))
    edge_feature_tensor = torch.stack(edge_feature_list, dim=0)

    return edge_feature_tensor


def construct_data_aug():
    pass
