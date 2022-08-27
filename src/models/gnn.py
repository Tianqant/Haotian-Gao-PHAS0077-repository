import torch
import numpy as np
import os
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from .model_template import ModelTemplate
from .utils import mse_loss, ce_loss, binary_ce_loss
from .sage_gnn import SAGEGNNEncoder
# from .sage_gnn_hetero import SAGEGNNEncoder_hetero
# from .GAT_gnn import GAT
# from .heteroGNN1 import HeteroGNN1
from .utils import RandomLinkSplit


OPTIMIZERS = {
    "sparse-adam": torch.optim.SparseAdam,
    "adam": torch.optim.Adam
}

LOSS = {
    "mse": mse_loss,
    "cross-entropy": ce_loss,
    "binary-cross-entropy": binary_ce_loss
}

ENCODERS = {
    "sage-conv": SAGEGNNEncoder,
    # "sage-conv-with-linear": SAGEGNNEncoder_hetero,
    # "gat": GAT,
    # "hetero-gnn-1": HeteroGNN1
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class EdgeDecoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout=0.0):
        super().__init__()
        self.dropout_fn = torch.nn.Dropout(dropout)

        self.lins = torch.nn.ModuleList()

        input_layer = Linear(2*input_channels, hidden_channels[0])
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
            x = self.dropout_fn(layer(x))
            x = F.leaky_relu(x)

        x = self.output_layer(x).sigmoid()
        return torch.cat([x, torch.ones_like(x) - x], dim=1)


class GNNModel(torch.nn.Module):
    def __init__(self, data, model_args):
        super().__init__()
        self.data = data.data if data else None
        encoder_name = model_args.pop("encoder_name")
        encoder_args = model_args.pop("encoder_args")

        decoder_args = model_args.pop("decoder_args")
        decoder_args["input_channels"] = encoder_args["out_channels"]

        self.encoder = ENCODERS[encoder_name](data, **encoder_args)
        if encoder_name == "sage-conv" or encoder_name == "gat":
            self.encoder = to_hetero(self.encoder, self.data.metadata(), aggr="max")

        self.decoder = EdgeDecoder(**decoder_args)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x, edge_label_index)


class GNNClf(ModelTemplate):
    def __init__(self, dataset, test_dataset, val_dataset=None, loss="mse", model_args=None, path=None):
        super().__init__("GNN Classifier")

        self.save_path = path
        self.data = dataset.data if dataset else None
        self.val_data = val_dataset.data if val_dataset else None
        self.test_data = test_dataset.data if test_dataset else None
        self.loss = LOSS[loss]

        optimizer_args = model_args.pop("optimizer")

        self.model = GNNModel(self.data, 
                              model_args).to(device)

        self.optimizer = OPTIMIZERS[optimizer_args["name"]](
            self.model.parameters(), **optimizer_args["args"])

        self.save_epochs = model_args.pop("save_epochs")
        self.batch_size = model_args.pop("batch_size")

        self.losses, self.val_losses = [], []
        self.accuracy, self.val_accuracy = [], []
        self.precision, self.val_precision = [], []
        self.recall, self.val_recall = [], []
        self.f1, self.val_f1 = [], []

        self.train_dataloader = NeighborLoader(
            self.data.data,
            directed=False,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors={key: [10] * 2 for key in self.data.data.edge_types},
            # Use a batch size of 128 for sampling training nodes
            batch_size=self.batch_size,
            input_nodes=("customer", self.data.data["customer"].node_index)
        )

    def describe(self):
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pt"))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt"),  map_location=torch.device('cpu')))

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
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_prec, train_rec, train_f1 = 0, 0, 0, 0, 0
            for n, train_data in enumerate(self.train_dataloader, start=1):
                self.model.train()
                self.optimizer.zero_grad()
                train_data.to(device)
                pred = self.model.forward(train_data.x_dict, train_data.edge_index_dict,
                                    train_data["customer", "purchases", "variant"].edge_index)

                target = train_data["customer", "purchases", "variant"].edge_label

                train_loss += self.loss(pred, target)

                if epoch % self.save_epochs == 0:

                    train_results = self.get_scores(pred, target, loss=self.loss)

                    train_acc += train_results['accuracy']
                    train_prec += train_results['precision']
                    train_rec += train_results['recall']
                    train_f1 += train_results['f1-score']

            train_loss /= n
            train_loss.backward()
            self.optimizer.step()

            if epoch % self.save_epochs == 0:
                val_results = self.validation()

                train_loss = float(train_loss.detach().cpu())
                val_loss = float(val_results['loss'].cpu())
                val_acc = val_results['accuracy']
                val_prec = val_results['precision']
                val_rec = val_results['recall']
                val_f1 = val_results['f1-score']

                self.losses.append(train_loss)
                self.val_losses.append(val_loss)

                self.accuracy.append(train_acc / n)
                self.val_accuracy.append(val_acc)

                self.precision.append(train_prec / n)
                self.val_precision.append(val_prec)

                self.recall.append(train_rec / n)
                self.val_recall.append(val_rec)

                self.f1.append(train_f1 / n)
                self.val_f1.append(val_f1)

                print(f"""Epoch {epoch}, Train CE loss: {self.losses[-1]:.3f}, 
                Train Accuracy: {100*self.accuracy[-1]:.2f}%,
                Validation CE loss: {self.val_losses[-1]:.3f},
                Validation Accuracy: {100*self.val_accuracy[-1]:.2f}%""")

                if len(self.val_losses) > 1 and self.val_losses[-1] < min(self.val_losses[:-1]):
                    self.save()

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        self.val_data.data.to(device)
        pred = self.model.forward(self.val_data.data.x_dict, self.val_data.data.edge_index_dict,
                            self.val_data.data["customer", "purchases", "variant"].edge_index)

        target = self.val_data.data["customer", "purchases", "variant"].edge_label

        scores = self.get_scores(pred, target, loss=self.loss)

        return scores

    @torch.no_grad()
    def test(self, data=False):
        self.model.eval()
        test_data = data if data else self.test_data[0]
        test_data.to(device)
        pred = self.model.forward(test_data.x_dict, test_data.edge_index_dict,
                            test_data["customer", "purchases", "variant"].edge_index)

        target = test_data["customer", "purchases", "variant"].edge_label

        scores = self.get_scores(pred, target, loss=self.loss)
        roc_scores = self.get_roc_scores(pred[:,0], target)
        scores["roc"] = roc_scores

        return scores