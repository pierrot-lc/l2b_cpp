"""Computes the metrics from a bunch of oucomes.
Uses the config of a training wandb instance.
"""
import sys
import os
from collections import defaultdict

import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics.read_scores import read_scores, list_files, parse_json
from src.metrics.metric import EnergyMetric


def energy(
        childs: torch.FloatTensor,
        parent: torch.FloatTensor,
        temperature: callable
    ) -> torch.FloatTensor:
    """Energy normalization.
    """
    t = temperature(parent)
    values = childs / t
    values = torch.softmax(values - values.max(), dim=-1)
    return values


def normal(childs: torch.FloatTensor) -> torch.FloatTensor:
    """Normal normalization.
    """
    if len(childs) == 1:
        return torch.FloatTensor([1])

    values = torch.softmax(
        - (childs - childs.mean()) / (childs.std() + 1e-9),
        dim=-1
    )
    return values


def normalize_energy(
        mcfix: torch.FloatTensor,
        sizes: torch.LongTensor,
        temperature: callable,
        norm: str,
    ) -> torch.FloatTensor:
    """Adapt the normalization between 'normal' and 'energy'.
    """
    normalized = torch.zeros(mcfix.shape, dtype=float)

    start_index, s = 0, sizes[0]
    childs = mcfix[start_index:start_index+s]
    parent = childs.mean()

    if norm == 'energy':
        normalized[start_index: start_index+s] = energy(childs, parent, temperature)
    elif norm == 'normal':
        normalized[start_index: start_index+s] = normal(childs)

    start_index += s
    parent = childs.max()

    for s in sizes[1:]:
        childs = mcfix[start_index: start_index+s]

        if norm == 'energy':
            normalized[start_index: start_index+s] = energy(childs, parent, temperature)
        elif norm == 'normal':
            normalized[start_index: start_index+s] = normal(childs)

        start_index += s
        parent = childs.min()

    return normalized


def energy_metrics(
        preds: dict,
        sizes: torch.LongTensor,
        temperature: callable,
        max_ecart: float,
        norm: str,
    ) -> dict:
    """Compute the metrics from the given predictions and parameters.
    """
    preds['mcfix_normalized'] = normalize_energy(preds['mcfix'], sizes, temperature, norm)
    metric = EnergyMetric(preds['iafix'], preds['mcfix_normalized'], sizes)

    metric.wrong_predictions()
    metric.DCG()
    metric.loss()
    metric.approx_accuracy(max_ecart)
    metric.precision()

    metrics = metric.metrics

    start_idx = 0
    cfix = preds['cfix']
    for s in sizes:
        pred = cfix[start_idx:start_idx+s]
        pred = torch.log(pred)
        pred = torch.softmax(
            (pred - pred.mean()) / (pred.std() + 1e-9)
        , dim=0)

        if len(pred) == 1:
            pred = torch.FloatTensor([1])

        cfix[start_idx:start_idx+s] = pred
        start_idx += s

    metric = EnergyMetric(preds['cfix'], preds['mcfix_normalized'], sizes)
    metric.y_pred = preds['cfix']

    metric.wrong_predictions()
    metric.DCG()
    acc = metric.approx_accuracy(max_ecart)
    metric.precision()

    for name, value in metric.metrics.items():
        metrics['Frac - ' + name] = value

    return metrics, acc


def update_df_metrics(
        df: pd.DataFrame,
        folder: str,
        config: dict,
    ) -> pd.DataFrame:
    """Append to the DataFrame the metrics from the given folder.
    Compute the metrics from the scores.
    """
    data = defaultdict(list)

    instance_name = os.path.basename(folder)
    files = list_files(folder)

    for path in files:
        preds, sizes = read_scores(path)
        if len(sizes) == 0:
            print('Empty file:', path)
            continue

        window = path.split('scores_')[1][:-len('.csv')]
        window = int(window)

        data['instance'].append(instance_name)
        data['window'].append(window)

        metrics, acc = energy_metrics(
            preds,
            sizes,
            config['temperature'],
            config['max_ecart'],
            config['norm']
        )
        for m in metrics:
            data[str(m)].append(float(metrics[m]))

        if 'big_attention' in path:
            continue
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(range(len(acc)), acc, '.')
            ax.set_title(path)
            fig.show()
            input('Press any to continue...')

    n_df = pd.DataFrame(data)
    df = pd.concat([df, n_df])
    return df


def from_experiment_dir(experiment_dir: str):
    """Compute the metrics from the scores saved in the
    multiple instances inside the experiment directory.
    """
    # List of all instance directories
    instance_folders = [
        os.path.join(experiment_dir, instance_folder)
        for instance_folder in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, instance_folder))
    ]

    # Read the config file
    json_path = os.path.join(experiment_dir, 'config.json')
    config = parse_json(json_path)
    if config['norm'] not in {'normal', 'energy'}:
        print(f'Error: unknown norm {config["norm"]}')
        sys.exit(0)

    df = pd.DataFrame()

    for folder in instance_folders:
        df = update_df_metrics(df, folder, config)

    csv_path = os.path.join(experiment_dir, 'metrics_gencol.csv')
    df.to_csv(csv_path, index=False)
