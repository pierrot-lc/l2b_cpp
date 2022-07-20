import inspect

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.train.utils import create_datasets, train
from src.train.residual import loss_batch, NORM, TEMPERATURE
from script.prepare_dagger import INSTANCES_PATH as DAGGER_INSTANCES_PATH


def init_config() -> dict:
    config = {
        'group': 'Residual',
        'norm': NORM,
        'temperature': None,
        'remove_fracval': False,
        'normalize_feats': False,

        'epochs': 1000,
        'batch size': 256,
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'T_0': 1000,
        'T_mult': 1,
        'lr_min': 1e-3,
        'weight_decay': 0.0,
        'l1_lambda': 0,
        'criterion': 'MAE',

        'model': 'MLP',

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 767976,
    }

    return config


def training_params(
    dagger: bool,
    jit_path: str,
    ) -> tuple:

    assert (not dagger) or (jit_path is not None)
    torch.randn((1, 5, 13))

    config = init_config()
    config['dagger'] = dagger

    torch.manual_seed(config['seed'])

    train_dataset, val_dataset = create_datasets(
        config['norm'],
        config['temperature'],
        remove_fracval = config['remove_fracval'],
        path = 'data/train' if not dagger else DAGGER_INSTANCES_PATH,
        test_size = 0.2,
        random_state = config['seed'],
        dagger = dagger or not config['normalize_feats'],
        residual = True
    )

    config['max_ecart'] = train_dataset.max_ecart
    config['prec_ref'] = train_dataset.prec_ref
    config['input_size'] = train_dataset.n_features
    config['forward_shape'] = (1, config['input_size'])  # [batch_size, n_childs, n_feats]
    config['args'] = (
        torch.rand(config['forward_shape'])
    )

    b_size = config['batch size']
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, collate_fn=list)
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=True, collate_fn=list)


    if dagger:
        model = torch.jit.load(jit_path)
    else:
        model = nn.Sequential(
            nn.Linear(config['input_size'], 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        model.input_size = config['input_size']


    config['optimizer'] = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        weight_decay=config['weight_decay'],
    )
    config['scheduler'] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        config['optimizer'],
        T_0=config['T_0'],
        T_mult=config['T_mult'],
        eta_min=config['lr_min'],
    )

    return model, train_loader, val_loader, config


def main(
        dagger: bool = False,
        jit_path: str = None,
    ):
    model, train_loader, val_loader, config = training_params(dagger, jit_path)

    print('\n\n\t\t\t\tSimple training module\n\n')

    summary(model, input_size=config['forward_shape'])

    print('\n')
    hp_cols = [
        'group',
        None,
        'epochs',
        'batch size',
        'device',
        None,
        'lr',
        'betas',
        'weight_decay',
        None,
        'T_0',
        'T_mult',
        'lr_min',
        None,
        'norm',
        'max_ecart',
        'prec_ref',
    ]
    for col in hp_cols:
        if col is None:
            print('')
        else:
            print(f'        [{col[:10]}]\t-\t{config[col]}'.expandtabs(25))

    ans = input('\n\nContinue ? [y/n]> ')
    if ans.lower() == 'y':
        train(model, config, train_loader, val_loader, loss_batch)
