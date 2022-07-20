import os

import pandas as pd

from src.data_utils.node import BBNode


def read_all_dfs(path):
    dfs = {}
    for file in os.listdir(path):
        dfs[file] = pd.read_csv(os.path.join(path, file))
    return dfs

def build_bbtree(df, filename):
    """
    Build the Branch and Bound tree for the given df.
    """
    col_id = {
        col_name: i
        for i, col_name in enumerate(df.columns)
    }

    features_name = [
        f for f in df.columns
        if f not in {'node_number', 'parent_node_number', 'value'}
    ]
    features_id = [col_id[f] for f in features_name]

    nodes = dict()  # node_id -> BBNode
    parent_map = dict()  # node_id -> parent_id

    # Build all BBNodes
    for row in df.values:
        node_id = row[col_id['node_number']]
        parent_id = row[col_id['parent_node_number']]
        value = row[col_id['value']]
        features = {
            f_name: row[f_id]
            for f_name, f_id in zip(features_name, features_id)
        }
        node = BBNode(filename, node_id, features, value)

        nodes[node_id] = node
        parent_map[node_id] = parent_id

    # Link BBNodes to their parent
    for node_id, node in nodes.items():
        parent_id = parent_map[node_id]
        if parent_id in nodes:
            parent_node = nodes[parent_id]
            parent_node.add_child(node)

    return nodes

def keep_parent_nodes(nodes):
    """
    Filter all nodes in the dictionary, to keep only
    the nodes that are parent to some other nodes.
    """
    parents_id = set(n_id for n_id, n in nodes.items()
                     if n.children_nodes)
    return {p_id: parent for p_id, parent in nodes.items()
            if p_id in parents_id}

def get_trees(path):
    dfs = read_all_dfs(path)

    trees = {
        f: build_bbtree(df, f)
        for f, df in dfs.items()
    }
    trees = {
        f: keep_parent_nodes(nodes)
        for f, nodes in trees.items()
    }
    return trees

def df_to_matrix(df):
    """
    Enleve les enfants qui n'ont pas assez de frères (fin de l'arbre en général).
    Transforme le reste des enfants en une matrice [stages, childs_size, features + value].
    """
    childs_size = max(df.groupby('parent_node_number').count()['node_number'])

    bad_parents = df.groupby('parent_node_number').count()['node_number'] != childs_size
    bad_parents = bad_parents[bad_parents == True].index
    to_remove = []
    for p in bad_parents:
        to_remove.extend(list(df[ df['parent_node_number'] == p ].index))
    df = df.drop(index=to_remove, axis='index')
    stages = len(df) // childs_size
    assert stages * childs_size == len(df)

    df = df.sort_values('parent_node_number', axis='index')
    features_columns = df.columns.drop(['node_number', 'parent_node_number', 'value'])
    features_columns = list(features_columns) + ['value']  # Place 'value' à la fin
    matrix = df[features_columns].values
    matrix = matrix.reshape((stages, childs_size, len(features_columns)))
    return matrix
