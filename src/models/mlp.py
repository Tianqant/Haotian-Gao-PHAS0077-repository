import torch
from torch.nn import Linear
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import os
import torch_geometric.transforms as T
import torch.nn.functional as F

from .utils import mse_loss, ce_loss, binary_ce_loss
from .model_template import ModelTemplate
# from .random_link_transform import RandomLinkSplit

LOSS = {
    "mse": mse_loss,
    "cross-entropy": ce_loss,
    "binary-cross-entropy": binary_ce_loss
}

OPTIMIZERS = {
    "sparse-adam": torch.optim.SparseAdam,
    "adam": torch.optim.Adam
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLPModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()

        self.lins = torch.nn.ModuleList()

        input_layer = Linear(input_channels, hidden_channels[0])
        self.lins.append(input_layer)

        if len(hidden_channels) > 1:
            for n in range(0, len(hidden_channels) - 1):
                lin = Linear(hidden_channels[n], hidden_channels[n+1])
                self.lins.append(lin)
            
        self.output_layer = Linear(hidden_channels[-1], 1)
        
    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([x_dict["customer"][row], x_dict["variant"][col]], dim=-1)
        
        for layer in self.lins:
            x = layer(x)
            x = torch.tanh(x)

        x = self.output_layer(x).sigmoid()
        return torch.cat([x, torch.ones_like(x) - x], dim=1)


class MLPClf(ModelTemplate):
    def __init__(self, dataset, test_dataset, val_dataset, loss="mse", model_args=None, path=None):
        super().__init__("MLP Classifier")

        self.save_path = path
        self.train_data = dataset.data.data if dataset else None
        self.val_data = val_dataset.data.data if val_dataset else None
        self.test_data = test_dataset.data.data if test_dataset else None
        self.loss = LOSS[loss]

        optimizer_args = model_args.pop("optimizer")

        self.model = MLPModel(**model_args).to(device)

        self.optimizer = OPTIMIZERS[optimizer_args["name"]](
            list(self.model.parameters()), **optimizer_args["args"])

        self.losses, self.val_losses = [], []
        self.accuracy, self.val_accuracy = [], []
        self.precision, self.val_precision = [], []
        self.recall, self.val_recall = [], []
        self.f1, self.val_f1 = [], []

    def describe(self):
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pt"))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=torch.device('cpu')))

    def get_train_results(self):
        scores = {
            "losses": self.losses,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1-score": self.f1
        }

        val_scores = {
            "losses": self.val_losses,
            "accuracy": self.val_accuracy,
            "precision": self.val_precision,
            "recall": self.val_recall,
            "f1-score": self.val_f1
        }

        return scores, val_scores

    def get_data(self):
        return self.train_data, self.val_data

    def decision_function(self, X):
        return self.model.decision_function(X)

    def train(self, epochs):
        train_data, val_data = self.get_data()

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model.forward(train_data.x_dict,
                                train_data["customer", "purchases", "variant"].edge_index)

            target = train_data["customer", "purchases", "variant"].edge_label

            loss = self.loss(pred, target)
            loss.backward()
            self.optimizer.step()

            train_results, val_results = self.get_training_scores()

            if epoch > 1 and float(val_results['loss'].cpu()) < min(self.val_losses):
                self.save()

            self.losses.append(float(loss.detach().cpu()))
            self.val_losses.append(float(val_results['loss'].cpu()))

            self.accuracy.append(train_results['accuracy'])
            self.val_accuracy.append(val_results['accuracy'])

            self.precision.append(train_results['precision'])
            self.val_precision.append(val_results['precision'])

            self.recall.append(train_results['recall'])
            self.val_recall.append(val_results['recall'])

            self.f1.append(train_results['f1-score'])
            self.val_f1.append(val_results['f1-score'])


            print(f"""Epoch {epoch}, Train CE loss: {train_results['loss']:.3f}, 
            Train Accuracy: {100*train_results['accuracy']:.2f}%,
            Validation CE loss: {val_results['loss']:.3f},
            Validation Accuracy: {100*val_results['accuracy']:.2f}%""")

    def get_training_scores(self):
        self.model.eval()
        train_data, val_data = self.get_data()

        target = train_data["customer", "purchases", "variant"].edge_label
        pred = self.model.forward(train_data.x_dict,
                                train_data["customer", "purchases", "variant"].edge_index)

        scores = self.get_scores(pred, target, loss=self.loss)

        val_scores = self.validation(val_data)

        return scores, val_scores

    @torch.no_grad()
    def validation(self, data):
        self.model.eval()
        pred = self.model.forward(data.x_dict,
                            data["customer", "purchases", "variant"].edge_index)

        target = data["customer", "purchases", "variant"].edge_label

        scores = self.get_scores(pred, target, loss=self.loss)

        return scores

    @torch.no_grad()
    def test(self, data=False):
        self.model.eval()
        test_data = data if data else self.test_data

        pred = self.model.forward(test_data.x_dict,
                            test_data["customer", "purchases", "variant"].edge_index)

        target = test_data["customer", "purchases", "variant"].edge_label

        scores = self.get_scores(pred, target, loss=self.loss)
        roc_scores = self.get_roc_scores(pred[:,0], target)
        scores["roc"] = roc_scores

        return scores