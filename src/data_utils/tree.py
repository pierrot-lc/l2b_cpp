"""Object representing a B&B tree.
"""
import scipy
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import torch
import torch.utils.data.dataset as dataset

from src.data_utils.node import BBNode

DUAL_COST_MIN_ID = 2
FRACVAL_ID = 4


class Tree:
    """
    Objet permettant de charger un arbre à partir d'une racine BBNode.
    On peut ensuite récupérer le ieme étage de l'arbre facilement.
    Les values sont normalisées 'on the fly', permettant de facilement
    tester plusieurs normalisations.
    """
    def __init__(
            self,
            root: BBNode,
            norm: str,
            temperature = None,
            remove_fracval: bool = False,
            features_normalisation: bool = True,
        ):
        """
        ---- args
        root: noeud racine de l'arbre.

        norm: type de normalisation, 'energy' ou 'minmax'.

        temperature: pour une normalisation de type 'energy', temperature doit être
        une fonction appelable, prenant en paramètre la valeur du noeud parent et retournant la
        valeur de la température.
        """
        norms = {'energy', 'minmax', 'normal'}
        if norm not in norms:
            raise RuntimeError(f"norm parameter '{norm}' not found. Expected one of {norms}.")

        if norm == 'energy' and temperature is None:
            raise RuntimeError(f"temperature argument '{temperature}', shoud be callable.")

        self.norm = norm
        self.columns_order = [c for c in sorted(root.features.keys())]
        self.euclidian_columns = []
        self.remove_fracval = remove_fracval
        self.features_normalisation = features_normalisation
        self.build_from_root(root)

        if norm == 'minmax':
            values = [self.root[1]]
            for _, v in self.stages:
                values.extend(v)
            self.min_v, self.max_v = min(values), max(values)
        elif norm == 'energy':
            self.temperature = temperature

    def build_from_root(self, root: BBNode):
        """
        Rassemble les features et values dans une liste d'étages.
        """
        self.root = (Tree.order_features(root), root.value)
        self.stages = []
        n_feat = len(self.root[0])
        current_node = root

        while current_node is not None:
            next_node = None
            n_childs = len(current_node.children_nodes)
            features = np.zeros((n_childs, n_feat))
            values = np.zeros(n_childs)
            for i, n in enumerate(current_node.children_nodes):
                features[i] = Tree.order_features(n)
                values[i] = n.value

                if n.children_nodes:  # Prochain parent pour l'itération suivante
                    next_node = i

            if next_node is not None:
                features[0], features[next_node] = features[next_node], features[0]
                values[0], values[next_node] = values[next_node], values[0]
                current_node = current_node.children_nodes[next_node]
            else:
                current_node = None  # Fin de l'arbre

            self.stages.append((features, values))

    def order_features(node: BBNode):
        """
        Retourne les features du node dans un array.
        Les features sont triées par ordre alphabétique (de leur nom).
        Permet d'assurer que les features des noeuds sont bien alignées en colonne.
        """
        return np.array([
            node.features[f] for f in sorted(node.features.keys())
        ])

    def register_euclidian(self, columns):
        """
        Sauvegardes les indices des colonnes données.

        Ces colonnes seront normalisées en live lors de l'accès au dataset.
        """
        self.euclidian_columns = [
            col_id for col_id, col in enumerate(self.columns_order)
            if col in columns
        ]

    def normalize_euclidian(features, col_id):
        # l2 = np.sqrt(np.power(features[:, col_id], 2).sum())
        # features[:, col_id] /= max(l2, 1)
        moy = np.mean(features[:, col_id], axis=0)
        std = np.std(features[:, col_id], axis=0)
        features[:, col_id] = (features[:, col_id] - moy) / (std + 1e-9)

    def register_log(self, columns):
        """
        Sauvegardes les indices des colonnes données.

        Ces colonnes seront normalisées en live lors de l'accès au dataset.
        """
        self.log_columns = [
            col_id for col_id, col in enumerate(self.columns_order)
            if col in columns
        ]

    def __len__(self):
        return len(self.stages)

    def __getitem__(self, i):
        """
        Retourne le ieme stage.
        Normalise les données selon le critère demandé à l'initialisation.
        """
        features, values = self.stages[i]
        features, values = features.copy(), values.copy()
        if self.norm == 'minmax':
            values = (values - self.min_v) / (self.max_v - self.min_v)
        elif self.norm == 'energy':
            parent_v = self.root[1] if i == 0 else self.stages[i-1][1][0]
            t = self.temperature(parent_v)
            values = values / t
            values = scipy.special.softmax(
                -(values - np.max(values)),
                axis=-1
            )  # softmax(x) == softmax(x - max(x))
        elif self.norm == 'normal':
            values = np.log(values)
            values = scipy.special.softmax(
                - (values - np.mean(values)) / (np.std(values) + 1e-9)
            )

        if self.features_normalisation:
            features[:, self.log_columns] = np.log(features[:, self.log_columns])
            features[:, DUAL_COST_MIN_ID] = np.log(np.abs(features[:, DUAL_COST_MIN_ID]) + 1)
            Tree.normalize_euclidian(features, self.euclidian_columns)

        if len(features) != 1:
            # Avoid computing softmax when there's only one childs (avoid 0/0)
            normal_fracval = np.log(features[:, FRACVAL_ID])
            normal_fracval = scipy.special.softmax(
                (normal_fracval - np.mean(normal_fracval)) / (np.std(normal_fracval) + 1e-9)
            )
        else:
            normal_fracval = np.ones((1, 1))

        features = np.append(features, normal_fracval.reshape(-1, 1), axis=1)


        if self.remove_fracval:
            features = np.delete(features, FRACVAL_ID, axis=1)

        return features, values


class DatasetTree(dataset.Dataset):
    """
    Mappe un index unique à chaque stage de chaque arbre.
    """
    def __init__(
            self,
            trees: list,
            euclidian_columns: list,
            log_columns: list
        ):
        """
        Args
        ----
            trees:              List of all trees in the dataset.
                Type is list[Tree].
            euclidian_columns:  All features that will be normalized
                euclidian norm.
                Type is list[str].
        """
        self.trees = trees
        self.total_stages = sum(len(t) for t in trees)

        # Pour trouver rapidement l'arbre à parcourir
        self.accumulator_trees = [0]
        for t in trees:
            self.accumulator_trees.append(self.accumulator_trees[-1] + len(t))
            t.register_euclidian(euclidian_columns)
            t.register_log(log_columns)

        self.remove_fracval = self.trees[0].remove_fracval
        self.init_ref()
        self.n_features = self.trees[0][0][0].shape[1]  # First tree -> first item -> values -> n_features

    def init_ref(self):
        """
        Initialise les valeurs de références pour les métriques de performance.

        max_ecart: écart médian entre le premier et le second noeud de chaque stage.
        """
        ecarts_max, ecarts_prec = [], []
        for tree in self.trees:
            for _, values in tree:
                if len(values) < 2:
                    continue

                values = sorted(values, reverse=True)
                ecarts_max.append(values[0] - values[1])
                ecarts_prec.append(values[0] - values[-1])

        if ecarts_prec == []:
            ecarts_prec = [0]
            ecarts_max = [0]

        self.max_ecart = np.median(ecarts_max)
        self.prec_ref = np.quantile(ecarts_prec, q=0.10)
        self.ecarts_prec = np.array(ecarts_prec)

    def print_stats(self):
        """
        Affiche la moyenne, mediane, écart-type et quelques quantiles des values.
        """
        val = []
        for tree in self.trees:
            for _, values in tree:
                val.extend(values)
        print('Moyenne:', np.mean(val))
        print('Ecart-type:', np.std(val))
        print('Mediane', np.median(val))
        for q in [0.1, 0.3, 0.7, 0.9]:
            print(f'{int(100*q)}% quantile:', np.quantile(val, q))

    def index_to_coords(self, index: int):
        """
        Retourne le numéro de l'arbre et le numéro du stage
        correspondant à l'index.

        Effectue une recherche par dichotomie sur l'accumulateur de stages.
        """
        bornes = [0, len(self.accumulator_trees) - 1]
        while bornes[1] - bornes[0] != 1:
            tree_id = (bornes[1] - bornes[0]) // 2
            if self.accumulator_trees[tree_id + bornes[0]] < index:
                bornes[0] = tree_id + bornes[0]
            else:
                bornes[1] = tree_id + bornes[0]

        if self.accumulator_trees[bornes[1]] == index:
            tree_id = bornes[1]
            stage_id = 0
        else:
            tree_id = bornes[0]
            stage_id = index - self.accumulator_trees[tree_id]
        return tree_id, stage_id

    def __len__(self):
        return self.total_stages

    def __getitem__(self, index: int):
        tree_id, stage_id = self.index_to_coords(index)
        features, values = self.trees[tree_id][stage_id]
        return torch.FloatTensor(features), torch.FloatTensor(values)


class DatasetResidual(DatasetTree):
    """
    Mappe un index unique à chaque stage de chaque arbre.

    La valeur de y est l'ecart dx a ajouter aux predictions
    de cfix pour combler l'erreur residuelle.
    """
    def __init__(
            self,
            trees: list,
            euclidian_columns: list,
            log_columns: list
        ):
        super().__init__(trees, euclidian_columns, log_columns)

    def __getitem__(self, index: int):
        features, values = super().__getitem__(index)
        y_cfix = features[:, -1]
        residual_error = torch.log(values) - torch.log(y_cfix)
        # residual_error = residual_error / (residual_error.abs().max() + 1e-5)
        return features, residual_error, values
