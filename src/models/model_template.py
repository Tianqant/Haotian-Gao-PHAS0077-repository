from abc import ABC, abstractstaticmethod
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score
import torch
import numpy as np

from .utils import mse_loss

LOSS = {
    "mse": mse_loss
}

class ModelTemplate(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractstaticmethod
    def describe(self):
        pass

    @abstractstaticmethod
    def train(self):
        pass

    @abstractstaticmethod
    def decision_function(self):
        pass

    def get_scores(self, y_pred, y_true, loss):
        scores = {}

        scores["loss"] = loss(y_pred, y_true)

        if isinstance(y_pred, torch.Tensor):
            y_pred = np.round(y_pred.detach().cpu().numpy()[:,1])
        else:
            y_pred = np.round(y_pred[:,1])

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        
        scores["accuracy"] = accuracy_score(y_pred, y_true)
        scores["precision"] = precision_score(y_pred, y_true)
        scores["recall"] = recall_score(y_pred, y_true)
        scores["f1-score"] = f1_score(y_pred, y_true)

        return scores

    def get_roc_scores(self, y_scores, y_true):
        scores = {}

        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.detach().cpu().numpy()

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        scores["tpr"], scores["fpr"] = tpr, fpr
        scores["auc"] = roc_auc

        return scores