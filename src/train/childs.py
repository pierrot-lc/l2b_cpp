"""Train a model that can output the values of all childs at once.
It takes the features of all childs.
"""
import torch
import torch.nn as nn

from src.metrics.metric import EnergyMetric, MinMaxMetric


NORM = 'normal'


def loss_batch(
        model: nn.Module,
        batch: list,
        device: str,
        norm: str,
        max_ecart: float
    ) -> tuple:
    """Compute metrics and loss of one batch.
    """
    total_nodes = sum(len(v) for _, v in batch)
    batch_size = len(batch)
    max_childs = max(len(v) for _, v in batch)
    n_feats = batch[0][0].shape[1]

    X = torch.zeros((batch_size, max_childs, n_feats), dtype=torch.float32)
    y = torch.zeros((batch_size, max_childs), dtype=torch.float32)
    fracval = torch.zeros((batch_size, max_childs), dtype=torch.float32)
    key_mask = torch.zeros((batch_size, max_childs), dtype=torch.bool)

    sizes = []
    # nodes are of shape [n_childs, n_feats]
    # values are of shape [n_childs]
    for batch_id, (nodes, values) in enumerate(batch):
        n_childs = len(values)

        X[batch_id, :n_childs] = nodes
        y[batch_id, :n_childs] = values
        fracval[batch_id, :n_childs] = nodes[:, -1]
        key_mask[batch_id, n_childs:] = torch.ones(max_childs - n_childs, dtype=torch.bool)

        sizes.append(n_childs)


    X, y = X.to(device), y.to(device)
    key_mask = key_mask.to(device)
    y_pred = model(X, key_mask)

    y_cpy_pred = torch.zeros(total_nodes).to(device)
    y_cpy_real = torch.zeros(total_nodes).to(device)
    fracval_cpy = torch.zeros(total_nodes).to(device)
    i = 0
    for batch_id, s in enumerate(sizes):
        y_cpy_real[i:i+s] = y[batch_id, :s]
        y_cpy_pred[i:i+s] = y_pred[batch_id, :s]
        fracval_cpy[i:i+s] = fracval[batch_id, :s]
        i += s

    y, y_pred = y_cpy_real, y_cpy_pred

    if norm == 'minmax':
        metric = MinMaxMetric(y_pred, y, sizes)
    elif norm == 'energy' or 'normal':
        metric = EnergyMetric(y_pred, y, sizes)
        metric.wrong_predictions()
        metric.DCG()

    metric.loss()
    metric.approx_accuracy(max_ecart)
    metric.precision()

    preds = {
        'y_real': metric.y_real,
        'y_pred': metric.y_pred,
    }
    return metric.metrics, preds, fracval_cpy
