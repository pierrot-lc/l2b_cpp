import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.train.utils import create_datasets, train
from src.train.childs import loss_batch, NORM
from src.data_utils.tree import DatasetTree
from src.models.attention_childs import AttentionChilds
from script.prepare_dagger import INSTANCES_PATH as DAGGER_INSTANCES_PATH


def init_config() -> dict:
    config = {
        'group': 'AttentionChilds',
        'norm': NORM,
        'temperature': None,
        'remove_fracval': False,
        'normalize_feats': False,

        'epochs': 100,
        'batch size': 64,
        'lr': 1e-3,
        'betas': (0.9, 0.99),
        'T_0': 100,
        'T_mult': 2,
        'lr_min': 1e-4,
        'weight_decay': 0,
        'criterion': 'crossentropy',

        'n_layers': 1,
        'dim_embedding': 5,
        'nhead': 1,
        'dim_feedforward_transformer': 10,
        'dropout': 0.10,

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
    )

    config['max_ecart'] = train_dataset.max_ecart
    config['prec_ref'] = train_dataset.prec_ref
    config['input_size'] = train_dataset.n_features
    config['forward_shape'] = (1, 5, config['input_size'])  # [batch_size, n_childs, n_feats]
    config['args'] = (
        torch.randn(config['forward_shape']),
        torch.zeros((1, 5), dtype=torch.bool),
    )

    b_size = config['batch size']
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, collate_fn=list)
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=True, collate_fn=list)

    if dagger:
        model = torch.jit.load(jit_path, map_location='cpu')
    else:
        model = AttentionChilds(
            config['input_size'],
            config['n_layers'],
            config['dim_embedding'],
            config['dropout'],
            config['nhead'],
            config['dim_feedforward_transformer'],
        )

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
    """Train a model where the configuration is
    given in the `init_config` function.

    If `dagger` is set to `True`, it will train the model
    in the `dagger/model.pt' file.
    In this case, the dataset is made of the output instances
    in the folder `dagger/instances`.
    """
    model, train_loader, val_loader, config = training_params(dagger, jit_path)

    print('\n\n\t\t\t\tChilds attention training module\n\n')

    summary(model,
        input_size = [
            config['forward_shape'],
            (1, 5)
        ]
    )

    print('\n')
    hp_cols = [
        'group',
        None,
        'epochs',
        'batch size',
        'device',
        'dagger',
        None,
        'lr',
        'betas',
        'weight_decay',
        None,
        'T_0',
        'T_mult',
        'lr_min',
        None,
        'input_size',
        'n_layers',
        'dim_embedding',
        'dropout',
        'nhead',
        'dim_feedforward_transformer',
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
