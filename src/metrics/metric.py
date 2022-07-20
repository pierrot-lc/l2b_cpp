"""Compute metrics from the predictions and the real values.
The metrics are defined here, and some of them depends on the values normalization.
"""
import torch
import torch.nn as nn


class Metric:
    """Base class for the metrics' computations.
    """
    def __init__(
            self,
            predictions: torch.FloatTensor,
            y_real: torch.FloatTensor,
            sizes: torch.LongTensor
        ):
        """
        Args
        ----
            predictions:    Brut predictions of the model (before softmax / sigmoid).
                Shape of [total_nodes, ].
            y_real:         Values provided by the strong branching heuristic. Should be normalized.
                Shape of [total_nodes, ].
            sizes:          Vector describing the number of stacked children in `y_real` and `predictions` vectors.
                Shape of [batch_size, ].
        """
        self.predictions = predictions
        self.y_real = y_real
        self.sizes = sizes
        self.metrics = dict()
        self.compute_y()

    def compute_y(self):
        """Should be implemented by the inherited class.
        It depends of the normalization of `y_real`.
        """
        self.y_pred = None
        self.stage_pred = None  # Choosen node at each stage (indexes)
        self.stage_real = None  # Best node at each stage (indexes)

    def precision(self):
        self.metrics['precision'] = torch.abs(self.y_real - self.y_pred).mean().item()

    def approx_accuracy(self, max_ecart: float):
        node_pred = torch.gather(self.y_real, 0, self.stage_pred)
        node_real = torch.gather(self.y_real, 0, self.stage_real)

        acc = torch.abs(node_pred - node_real) < max_ecart
        not_best = (node_pred != node_real) & acc  # Pas le meilleur noeud, mais validé
        not_best = not_best.sum() / acc.sum()
        acc = acc.float().mean()

        self.metrics['approx acc'] = acc.cpu().item()
        self.metrics['good but not best'] = not_best.cpu().item()
        self.metrics['top-1 acc'] = (node_pred == node_real).float().mean().cpu().item()
        accs = (node_pred == node_real).float().cpu().numpy()
        return accs

    def DCG(self):
        node_real = torch.gather(self.y_real, 0, self.stage_real)
        node_pred = torch.gather(self.y_pred, 0, self.stage_pred)

        DCG = (self.y_pred * self.y_real).sum().cpu().item()
        self.metrics['nDCG'] = DCG / self.y_real.pow(2).sum().cpu().item()


class MinMaxMetric(Metric):
    """Metrics for the normalization min-max.
    """
    def __init__(
            self,
            predictions: torch.FloatTensor,
            y_real: torch.FloatTensor,
            sizes: torch.LongTensor
        ):
        Metric.__init__(self, predictions, y_real, sizes)

    def compute_y(self):
        self.y_pred = torch.sigmoid(self.predictions)

        self.stage_pred = torch.zeros(len(self.sizes), dtype=torch.int64).to(self.y_real.device)
        self.stage_real = torch.zeros(len(self.sizes), dtype=torch.int64).to(self.y_real.device)
        i = 0
        for j, s in enumerate(self.sizes):
            _, min_idx = self.y_pred[i:i+s].min(dim=0)
            self.stage_pred[j] = min_idx + i

            _, min_idx = self.y_real[i:i+s].min(dim=0)
            self.stage_real[j] = min_idx + i

            i += s

    def loss(self):
        loss = - self.y_real * nn.functional.logsigmoid(self.predictions + 1e-6) \
            - (1 - self.y_real) * nn.functional.logsigmoid(1 - self.predictions + 1e-6)
        loss = loss.sum()

        self.metrics['loss'] = loss


class EnergyMetric(Metric):
    """Metrics for the normalization by energy.
    """
    def __init__(
            self,
            predictions: torch.FloatTensor,
            y_real: torch.FloatTensor,
            sizes: torch.LongTensor
        ):
        Metric.__init__(self, predictions, y_real, sizes)

    def compute_y(self):
        self.y_pred = torch.zeros(self.predictions.shape).to(self.predictions.device)
        i = 0
        for s in self.sizes:
            self.y_pred[i:i+s] = torch.softmax(self.predictions[i:i+s], dim=0)
            i += s

        self.stage_pred = torch.zeros(len(self.sizes), dtype=torch.int64).to(self.y_real.device)
        self.stage_real = torch.zeros(len(self.sizes), dtype=torch.int64).to(self.y_real.device)
        i = 0
        for j, s in enumerate(self.sizes):
            _, max_idx = self.y_pred[i:i+s].max(dim=0)
            self.stage_pred[j] = max_idx + i

            _, max_idx = self.y_real[i:i+s].max(dim=0)
            self.stage_real[j] = max_idx + i

            i += s

    def loss(self):
        loss, i = 0, 0
        for s in self.sizes:
            loss_stage = -self.y_real[i:i+s] * nn.functional.log_softmax(self.predictions[i:i+s] + 1e-9, dim=0)
            loss += loss_stage.sum()
            i += s

        self.metrics['loss'] = loss / sum(self.sizes)

    def wrong_predictions(self, max_score: float=0.15):
        """
        Pourcentage de mauvaises prédictions. On considère qu'une prédiction
        est mauvaise lorsque la valeur réelle du noeud choisit par le modèle
        est inférieure à `max_score` (défaut 15%).
        """
        node_pred = torch.gather(self.y_real, 0, self.stage_pred)
        wrongs = node_pred < max_score
        wrongs = wrongs.float().mean().cpu().item()
        self.metrics['wrong_predictions'] = wrongs


class ResidualMetric(EnergyMetric):
    """Metrics for the normalization by energy.
    """
    def __init__(
            self,
            predictions: torch.FloatTensor,
            dx_real: torch.FloatTensor,
            y_real: torch.FloatTensor,
            sizes: torch.LongTensor,
            fracval: torch.FloatTensor,
        ):
        self.fracval = fracval.to(predictions.device)
        self.dx_real = dx_real
        super().__init__(predictions, y_real, sizes)

    def compute_y(self):
        self.y_pred = torch.zeros(self.predictions.shape).to(self.predictions.device)
        # self.y_real = torch.zeros(self.predictions.shape).to(self.predictions.device)

        i = 0
        for s in self.sizes:
            if s == 1:
                self.y_pred[i] = 1
                # self.y_real[i] = 1
            else:
                x_cfix = torch.log(self.fracval[i:i+s])
                x_cfix = (x_cfix - x_cfix.mean()) / (x_cfix.std() + 1e-5)
                self.y_pred[i:i+s] = torch.softmax(x_cfix + self.predictions[i:i+s], dim=0)
                # self.y_real[i:i+s] = torch.softmax(x_cfix + self.dx_real[i:i+s], dim=0)

            i += s

        self.stage_pred = torch.zeros(len(self.sizes), dtype=torch.int64).to(self.y_real.device)
        self.stage_real = torch.zeros(len(self.sizes), dtype=torch.int64).to(self.y_real.device)
        i = 0
        for j, s in enumerate(self.sizes):
            _, max_idx = self.y_pred[i:i+s].max(dim=0)
            self.stage_pred[j] = max_idx + i

            _, max_idx = self.y_real[i:i+s].max(dim=0)
            self.stage_real[j] = max_idx + i

            i += s

    def loss(self):
        """
        loss, i = 0, 0
        for s in self.sizes:
            loss_stage = -self.y_real[i:i+s] * torch.log(self.y_pred[i:i+s] + 1e-9)
            loss += loss_stage.sum()
            i += s
        """
        loss = (self.predictions - self.dx_real).abs().sum()
        self.metrics['loss'] = loss / sum(self.sizes)
