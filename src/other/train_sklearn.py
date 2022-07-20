import numpy as np
import torch

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR

from src.train.utils import create_datasets
from src.metrics.metric import ResidualMetric
from src.data_utils.tree import FRACVAL_ID


def init_config():
    return {
        'seed': 999,
    }


def add_brothers_features(X: np.ndarray) -> np.ndarray:
    n_bros, n_feats = X.shape
    select_bros = np.arange(n_bros - 1)
    X_new = np.zeros((n_bros, n_bros * n_feats), dtype=float)
    for node_id in range(n_bros):
        node_feats = X[node_id]

        select_bros = np.random.permutation(select_bros)
        bros_feats = X[np.arange(n_bros) != node_id][select_bros].flatten()

        X_new[node_id, :n_feats] = node_feats.copy()
        X_new[node_id, n_feats:] = bros_feats.copy()

    return X_new


def get_datasets(config: dict) -> dict:
    train_dataset, val_dataset = create_datasets(
        'normal',
        None,
        remove_fracval = False,
        # path = 'data/train',
        path = 'dagger/instances',
        test_size = 0.2,
        random_state = config['seed'],
        dagger = True,
        residual = True
    )
    config['max_ecart'] = train_dataset.max_ecart

    data = {}
    for dataset, data_type in [(train_dataset, 'train'), (val_dataset, 'val')]:
        X, y = [], []
        dx, fracval = [], []
        sizes = []
        for features, delta, values in dataset:
            if len(features) != 5:
                continue  # We only take 5-nodes stages

            X.append(add_brothers_features(features.numpy()))
            dx.append(delta.numpy())
            y.append(values.numpy())
            fracval.append(features[:, FRACVAL_ID].numpy())
            sizes.append(len(features))

        data[data_type] = {
            'X': np.concatenate(X, axis=0),
            'dx': np.concatenate(dx, axis=0),
            'y': np.concatenate(y, axis=0),
            'fracval': np.concatenate(fracval, axis=0),
            'sizes': sizes,
        }

    return data


def train_model(X_train, dx_train, config):
    """
    model = DecisionTreeRegressor(
        criterion='absolute_error',
        splitter='best',
        max_depth=None,
        min_samples_leaf=4,
    )
    """
    model = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.1,
        max_depth=4,
        random_state=config['seed'],
        loss='squared_error',
    )
    """
    model = GaussianProcessRegressor()
    """
    """
    model = Ridge(alpha=1)
    """
    """
    model = ExtraTreeRegressor(
        random_state=config['seed'],
    )
    """

    model.fit(X_train, dx_train)
    return model


def eval_model(model, X, dx_real, y, fracval, sizes, config):
    y_pred = model.predict(X)
    # y_pred = np.zeros_like(y_pred)
    metric = ResidualMetric(
        torch.FloatTensor(y_pred),
        torch.FloatTensor(dx_real),
        torch.FloatTensor(y),
        sizes,
        torch.FloatTensor(fracval)
    )

    metric.loss()
    metric.approx_accuracy(config['max_ecart'])
    metric.DCG()
    metric.wrong_predictions()
    metric.precision()

    return metric.metrics
