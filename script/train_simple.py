import inspect

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.train.utils import create_datasets, train
from src.train.simple import loss_batch, NORM, TEMPERATURE
from src.data_utils.tree import DatasetTree
from src.models.mlp import MLP
from script.prepare_dagger import INSTANCES_PATH as DAGGER_INSTANCES_PATH


def init_config() -> dict:
    config = {
        'group': 'MLP - pretraining for DAGGER',
        'norm': NORM,
        'temperature': TEMPERATURE,
        'remove_fracval': False,

        'epochs': 5,
        'batch size': 64,
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'step_lr': 15,
        'gamma': 0.5,
        'weight_decay': 0.001,
        'criterion': 'crossentropy',

        'model': 'MLP',
        'n_layers': 5,
        'n_hidden': 100,
        'dropout': 0.3,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 767976,
    }

    return config


def training_params(
    dagger: bool,
    jit_path: str,
    ) -> tuple:

    assert (not dagger) or (jit_path is not None)

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
        dagger = dagger
    )

    config['max_ecart'] = train_dataset.max_ecart
    config['prec_ref'] = train_dataset.prec_ref
    config['input_size'] = train_dataset.n_features
    config['forward_shape'] = (1, config['input_size'])  # [batch_size, n_childs, n_feats]
    config['args'] = (
        torch.rand(config['forward_shape']).to(config['device'])
    )

    b_size = config['batch size']
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, collate_fn=list)
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=True, collate_fn=list)


    if dagger:
        model = torch.jit_load(jit_path)
    else:
        model = MLP(config['input_size'], config['n_layers'], config['n_hidden'], config['dropout'])
    
    
    config['optimizer'] = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        weight_decay=config['weight_decay'],
    )
    config['scheduler'] = optim.lr_scheduler.StepLR(
        config['optimizer'],
        step_size=config['step_lr'],
        gamma=config['gamma']
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
