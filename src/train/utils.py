import os
import shutil
from collections import defaultdict
from itertools import chain

import scipy
import numpy as np
import pandas as pd
import wandb as wb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

from src.data_utils.node import BBNode
from src.data_utils.utils import EUCLIDIAN_COLUMNS, LOG_COLUMNS, parse_tree_name
from src.metrics.metric import EnergyMetric
from src.data_utils.read_data import get_trees
from src.data_utils.tree import Tree, DatasetTree, DatasetResidual, FRACVAL_ID


def gather_instances(
        path: str,
        norm: str,
        temperature,
        remove_fracval: bool,
        dagger: bool
    ) -> list:
    """Gather the trees from the instances in the path.
    """
    instances = []
    trees = get_trees(path)

    for tree in trees.values():
        if tree == {}:
            continue

        root = tree[min(tree.keys())]
        instances.append(
            Tree(
                root,
                norm,
                temperature,
                remove_fracval=remove_fracval,
                features_normalisation=not dagger,
            )
        )

    return instances


def create_datasets(
        norm: str,
        temperature,
        remove_fracval: bool,
        path: str = 'data/train',
        test_size: float = 0.2,
        random_state: int = 99,
        dagger: bool = False,
        residual: bool = False,
    ) -> tuple:
    """Create the training and validation dataset.
    """
    train_instances = gather_instances(path, norm, temperature, remove_fracval, dagger)

    train_dataset, val_dataset = train_test_split(
        train_instances,
        test_size=test_size,
        random_state=random_state,
    )

    dataset_class = DatasetResidual if residual else DatasetTree
    train_dataset = dataset_class(train_dataset, EUCLIDIAN_COLUMNS, LOG_COLUMNS)
    val_dataset = dataset_class(val_dataset, EUCLIDIAN_COLUMNS, LOG_COLUMNS)

    return train_dataset, val_dataset


def eval_fractionnaire(dataset, config):
    """
    Evalue les métriques de la policy donnée par la meilleure valeur
    fractionnaire de chaque noeud.
    """
    predictions, y_real = [], []
    sizes = []
    for features, values in dataset:
        pred = torch.log(features[:, FRACVAL_ID])
        pred = torch.softmax(
            (pred - pred.mean()) / (pred.std() + 1e-9)
        , dim=0)

        if len(pred) == 1:
            pred = torch.FloatTensor([1])

        predictions.append(pred)
        y_real.append(values)
        sizes.append(len(values))

    predictions, y_real = torch.cat(predictions), torch.cat(y_real)
    metric = EnergyMetric(predictions, y_real, sizes)
    metric.y_pred = predictions

    metric.approx_accuracy(config['max_ecart'])
    metric.wrong_predictions()
    metric.DCG()
    metric.precision()
    metric.metrics['loss'] = - (y_real * torch.log(predictions + 1e-5)).mean()

    for name, value in metric.metrics.items():
        config['Frac - ' + name] = value


def eval_instance(
        model: nn.Module,
        tree: BBNode,
        batch_size: int,
        name: str,
        max_ecart: float,
        device: str,
        norm: str,
        loss_batch: callable,
    ) -> dict:
    """Evaluate the given BBNode (an instance).
    Return the metrics.
    """
    dataset = DatasetTree([tree], EUCLIDIAN_COLUMNS, LOG_COLUMNS)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=list)
    metrics = eval_dataset(model, dataloader, device, norm, max_ecart, loss_batch)
    return metrics


def eval_dataset(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        norm: str,
        max_ecart: float,
        loss_batch: callable,
    ) -> dict:
    """Evaluate the given dataset (contained in the dataloader).
    Use the loss_batch function as a parameter to compute the metrics.
    """
    model.to(device)
    model.eval()
    metrics = defaultdict(list)
    values_real, values_pred, fracval = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch_metrics, values, frac = loss_batch(model, batch, device, norm, max_ecart)

            values_real.extend(values['y_real'].tolist())
            values_pred.extend(values['y_pred'].tolist())
            fracval.extend(frac.tolist())

            batch_metrics['loss'] = batch_metrics['loss'].item()
            for name, value in batch_metrics.items():
                metrics[name].append(value)


    for name, values in metrics.items():
        metrics[name] = np.mean(values)

    # Create the plot distribution
    fig = plt.figure(figsize=(12, 8))
    ax = fig.subplots(1, 1)
    ax.hist(values_real, bins=100, alpha=1.0, label='Real')
    ax.hist(values_pred, bins=100, alpha=0.7, label='Predicted')
    ax.hist(fracval, bins=100, alpha=0.5, label='Frac value')
    ax.set_title('Values distribution')
    ax.legend()
    metrics['Values distribution'] = wb.Image(fig)

    plt.close(fig)  # Free memory

    return metrics


def save_model(model: nn.Module, config: dict):
    """Save the model to the 'artifacts' directory.
    The model is jit saved, so it can be used in GENCOL.
    """
    # Create the directoriy where the artifacts will be saved
    dir_path = os.path.join('artifacts', config['group'])
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)
    model_path = os.path.join(dir_path, 'model.pt')
    model.to('cpu')

    traced_script_module = torch.jit.trace(model, config["args"])
    traced_script_module.save(model_path)
    model.to(config['device'])


def train(
        model: nn.Module,
        config: dict,
        data_train: DataLoader,
        data_val: DataLoader,
        loss_batch: callable,
    ):
    """Train a simple model.
    """
    # Load les configs
    optimizer, scheduler = config['optimizer'], config['scheduler']
    norm, device = config['norm'], config['device']
    max_ecart, prec_ref = config['max_ecart'], config['prec_ref']
    model.to(device)
    best_loss = 100

    if not data_train.dataset.remove_fracval:
        # Ajoute les métriques de la policy fractionnaire aux configs
        eval_fractionnaire(data_train.dataset, config)

    with wb.init(
            project='Strong Branching - new GENCOL',
            entity='strongbranching',
            group=config['group'],
            config=config,
            save_code=True,
        ):
        print('Training the model...')
        for _ in tqdm(range(config['epochs'])):
            model.train()
            for batch in data_train:
                optimizer.zero_grad()
                metrics, _, _ = loss_batch(model, batch, device, norm, max_ecart)
                metrics['loss'].backward()
                optimizer.step()

            metrics_train = eval_dataset(model, data_train, device, norm, max_ecart, loss_batch)
            metrics_val = eval_dataset(model, data_val, device, norm, max_ecart, loss_batch)

            results = {}
            for name, value in metrics_train.items():
                results['Train - ' + name] = value
            results['Train - precision prct'] = np.mean(metrics_train['precision'] < data_train.dataset.ecarts_prec)
            results['Train - precision ref'] = prec_ref / metrics_train['precision']

            for name, value in metrics_val.items():
                results['Validation - ' + name] = value
            results['Validation - precision prct'] = np.mean(metrics_val['precision'] < data_val.dataset.ecarts_prec)
            results['Validation - precision ref'] = prec_ref / metrics_val['precision']

            results['lr'] = scheduler.get_last_lr()[-1]

            wb.log(results)

            scheduler.step()

            # Save the model if it has the best validation loss
            if best_loss > metrics_val['loss']:
                save_model(model, config)
                best_loss = metrics_val['loss']


        # Create the directoriy where the artifacts will be saved
        dir_path = os.path.join('artifacts', config['group'])
        model_path = os.path.join(dir_path, 'model.pt')
        metrics_path = os.path.join(dir_path, 'metrics.csv')
        print(f'Artifacts are saved into "{dir_path}"')

        model = torch.jit.load(model_path, map_location='cpu')
        model = model.to(device)

        path_to_dagger_models = './dagger/model/'
        if os.path.exists(path_to_dagger_models) and config['dagger']:
            model.save(os.path.join(path_to_dagger_models, 'model.pt'))

        print('Evaluating the final metrics...')
        path = 'data/train/'
        trees_train = get_trees(path)

        path = 'data/test/'
        trees_test = get_trees(path)

        list_trees = [(trees_train, config['normalize_feats']), (trees_test, config['normalize_feats'])]

        trees_dagger = None # Créer la variable pour éviter une erreur si elle n'existe pas à la ligne 272 (?)
        if config['dagger']:
            path = './dagger/instances/'
            trees_dagger = get_trees(path)
            list_trees.append((trees_dagger, False))

        model_metrics = defaultdict(list)
        for trees, normalisation in list_trees:
            train_instance = trees == trees_train
            dagger_instance = trees == trees_dagger
            for name, nodes in tqdm(trees.items()):
                if nodes == {}:
                    continue

                root = nodes[min(nodes.keys())]
                tree = Tree(root, norm, temperature=config['temperature'], remove_fracval=config['remove_fracval'], features_normalisation=normalisation)

                parsed_name = parse_tree_name(name=name, dagger=not normalisation)
                metrics = eval_instance(model, tree, config['batch size'], name, max_ecart, 'cpu', norm, loss_batch)

                for name, value in chain(parsed_name.items(), metrics.items()):
                    model_metrics[name].append(value)
                model_metrics['train_instance'].append(train_instance)
                model_metrics['dagger_instance'].append(dagger_instance)

        df = pd.DataFrame(model_metrics)
        df.to_csv(
            metrics_path,
            index=False,
            sep=','
        )

        metrics_to_save = [
            'top-1 acc',
            'wrong_predictions',
            'nDCG',
            'loss',
        ]
        train_metrics = df[ df['train_instance'] ]
        train_metrics = {
            'Final train ' + metric_name: train_metrics[metric_name].mean()
            for metric_name in metrics_to_save
        }
        test_metrics = df[ (~df['train_instance']) & (~df['dagger_instance']) ]
        test_metrics = {
            'Final test ' + metric_name: test_metrics[metric_name].mean()
            for metric_name in metrics_to_save
        }

        wb.log(train_metrics | test_metrics)

        # Artifacts
        metrics_artifact = wb.Artifact('metrics', type='metrics')
        metrics_artifact.add_file(metrics_path)
        wb.log_artifact(metrics_artifact)

        model_artifact = wb.Artifact('model', type='model')
        model_artifact.add_file(model_path)
        wb.log_artifact(model_artifact)

    print('Done!')
