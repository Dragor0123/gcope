import torch
from torch_geometric.nn import global_add_pool, GINConv
from fastargs.decorators import param
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_conv_layers, dropout):
        super(GIN, self).__init__()
        self.global_pool = global_add_pool
        self.layer_num = num_conv_layers
        self.dropout = dropout 
        self.hidden_dim = hidden

        self.layers = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            mlp = Seq(
                Linear(hidden, hidden),
                BatchNorm1d(hidden),
                ReLU(),
                Linear(hidden, hidden)
            )
            self.layers.append(GINConv(mlp))

        self.t1 = Linear(num_features, hidden)
        self.t2 = Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.t1.weight)
        torch.nn.init.zeros_(self.t1.bias)
        torch.nn.init.xavier_uniform_(self.t2.weight)
        torch.nn.init.zeros_(self.t2.bias)
        for layer in self.layers:
            for module in layer.nn:
                if isinstance(module, Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    @param('general.reconstruct')
    def forward(self, data, reconstruct):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch

        h = torch.dropout(x, p=self.dropout, train=self.training)
        h = torch.relu(self.t1(h))
        h = torch.dropout(h, p=self.dropout, train=self.training)

        for conv in self.layers:
            h = conv(h, edge_index)
            h = torch.dropout(h, p=self.dropout, train=self.training)

        h = self.t2(h)
        graph_emb = self.global_pool(h, batch)

        if reconstruct == 0.0:
            return graph_emb
        else:
            return graph_emb, h

from fastargs.decorators import param

def get_model(num_features, hid_dim=128, num_conv_layers=2, dropout=0.2):
    return GIN(num_features, hid_dim, num_conv_layers, dropout)
#@param('model.backbone.gin.dropout')
#def get_model(num_features, hid_dim, num_conv_layers, dropout):
#    return GIN(num_features, hid_dim, num_conv_layers, dropout)
