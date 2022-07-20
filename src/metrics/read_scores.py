"""Read the outcomes of GENCOL, and parse some output files.
"""
import os
import json

import torch
import pandas as pd


def parse_json(json_path: str) -> dict:
    """Return the parsed config file.
    """
    with open(json_path, 'r') as json_file:
        config = json.load(json_file)

    max_ecart = config['max_ecart']['value']
    max_ecart = float(max_ecart)

    temperature = config['temperature']['value']
    if temperature is not None:
        temperature = temperature.split(' = ')[1]
        temperature = eval(temperature)

    return {
        'max_ecart': max_ecart,
        'temperature': temperature,
        'norm': config['norm']['value'],
    }

def file_to_list(path: str) -> dict:
    """Read the scores of a file and
    parse it into a dictionnary.
    """
    all_scores = {
        'mcfix': [],
        'iafix': [],
        'cfix': [],
    }

    df = pd.read_csv(path)
    for parent_id, child_nodes in df.groupby('parent_node_number'):
        all_scores['mcfix'].append(child_nodes['value'].values)
        all_scores['iafix'].append(child_nodes['predicted'].values)
        all_scores['cfix'].append(child_nodes['frac_val'].values)

    return all_scores


def list_to_preds(scores: dict) -> tuple:
    """Transform the scores from lists to tensors.
    Also compute the sizes tensor.
    """
    mcfix, cfix, iafix = [], [], []
    sizes = []

    for m, c, i in zip(scores['mcfix'], scores['cfix'], scores['iafix']):
        mcfix.extend(m)
        cfix.extend(c)
        iafix.extend(i)

        sizes.append(len(m))

    preds = {
        'mcfix': torch.FloatTensor(mcfix),
        'cfix': torch.FloatTensor(cfix),
        'iafix': torch.FloatTensor(iafix),
    }
    return preds, torch.LongTensor(sizes)


def read_scores(path: str) -> tuple:
    """Parse the scores' files.
    """
    scores = file_to_list(path)
    preds, sizes = list_to_preds(scores)
    return preds, sizes


def list_files(path_to_dir) -> list:
    """List the files in the directory and
    return the path to all the files in the form of `scores_*.csv`.
    """
    paths = [
        os.path.join(path_to_dir, filename)
        for filename in os.listdir(path_to_dir)
        if filename.startswith('scores_') and filename.endswith('.csv')
    ]

    return paths
