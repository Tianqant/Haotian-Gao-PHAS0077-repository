import torch
from torch_geometric.nn import SAGEConv, Linear
import torch.nn.functional as F

class SAGEGNNEncoder(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, dropout=0.0, normalize=False):
        super().__init__()
        self.dropout_fn = torch.nn.Dropout(dropout)    

        self.convs = torch.nn.ModuleList()

        for hidden_channel in hidden_channels:
            self.convs.append(SAGEConv((-1, -1), hidden_channel, normalize=normalize, aggr="max"))

        self.conv2 = SAGEConv((-1, -1), out_channels, normalize=normalize, aggr="max")
        
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.dropout_fn(conv(x, edge_index))
            x = F.leaky_relu(x)

        x = self.dropout_fn(self.conv2(x, edge_index))
        x = F.leaky_relu(x)
        return x