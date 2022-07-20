"""Train a linear residual model.
It takes only the features of one node for each node's prediction.
"""
import torch
import torch.nn as nn

from src.metrics.metric import ResidualMetric
from src.data_utils.tree import FRACVAL_ID


NORM = 'normal'
# TEMPERATURE = lambda value: 10**(1.5)  # Temperature statique
TEMPERATURE = None

def loss_batch(
        model: nn.Module,
        batch: list,
        device: str,
        norm: str,
        max_ecart: float
    ) -> dict:
    """Compute metrics and loss of one batch.
    """
    total_nodes = sum(len(v) for _, _, v in batch)
    batch_size = len(batch)
    n_feats = batch[0][0].shape[1]

    X = torch.zeros((total_nodes, n_feats), dtype=torch.float32)
    y = torch.zeros(total_nodes, dtype=torch.float32)
    dx_real = torch.zeros(total_nodes, dtype=torch.float32)
    fracval_normal = torch.zeros(total_nodes, dtype=torch.float32)
    fracval = torch.zeros(total_nodes, dtype=torch.float32)

    sizes, i = [], 0
    for nodes, dx, values in batch:
        X[i:i+len(nodes)] = nodes
        dx_real[i:i+len(dx)] = dx
        y[i:i+len(values)] = values
        fracval_normal[i:i+len(values)] = nodes[:, -1]
        fracval[i:i+len(values)] = nodes[:, FRACVAL_ID]

        i += len(values)
        sizes.append(len(values))

    X, y = X.to(device), y.to(device)
    dx_real = dx_real.to(device)
    dx_pred = model(X).view(-1)

    metric = ResidualMetric(dx_pred, dx_real, y, sizes, fracval)
    metric.wrong_predictions()
    metric.DCG()
    metric.loss()
    metric.approx_accuracy(max_ecart)
    metric.precision()

    preds = {
        'y_real': metric.y_real,
        'y_pred': metric.y_pred,
    }

    metric = metric.metrics

    """
    # L1 norm
    l1_norm = sum(
        p.abs().sum()
        for p in model.parameters()
    )
    metric['loss'] = metric['loss'] + 0 *  l1_norm
    """

    return metric, preds, fracval_normal
