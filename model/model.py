import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool


def feed_forward(input_dim, num_hidden_layers, hidden_dim, output_dim, activation=nn.ReLU):
    dims = [input_dim] + [hidden_dim] * num_hidden_layers
    layers = []
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class MPNNLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        node_embedding_dim,
        edge_embedding_dim,
        edge_num_layers,
        edge_hidden_dim,
        activation=nn.ReLU,
    ):
        super().__init__(aggr="sum")
        self.edge_network = feed_forward(
            input_dim=edge_embedding_dim,
            num_hidden_layers=edge_num_layers,
            hidden_dim=edge_hidden_dim,
            output_dim=node_embedding_dim**2,
            activation=activation,
        )
        self.cell = nn.GRUCell(input_size=in_channels, hidden_size=out_channels)

    def forward(self, x, edge_index, edge_attr):

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):

        edge_embeddings = self.edge_network(edge_attr.float())

        d = int(edge_embeddings.shape[1] ** 0.5)

        edge_embeddings = edge_embeddings.view(-1, d, d)  # Reshape to [batch_size, d, d]

        x_j = x_j.unsqueeze(2)

        result = torch.bmm(edge_embeddings, x_j)

        return result.squeeze()

    def update(self, aggr_out, x):

        return self.cell(aggr_out, x)


class MPNN(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        num_propagation_steps,
        node_embedding_dim,
        edge_num_layers,
        edge_embedding_dim,
        edge_hidden_dim,
    ):
        super(MPNN, self).__init__()

        self.node_embedding = nn.Linear(node_dim, node_embedding_dim)
        self.edge_embedding = nn.Linear(edge_dim, edge_embedding_dim)

        self.mpnn_layers = nn.ModuleList(
            [
                MPNNLayer(
                    in_channels=node_embedding_dim,
                    out_channels=node_embedding_dim,
                    node_embedding_dim=node_embedding_dim,
                    edge_embedding_dim=edge_embedding_dim,
                    edge_num_layers=edge_num_layers,
                    edge_hidden_dim=edge_hidden_dim,
                )
                for _ in range(num_propagation_steps)
            ]
        )

        self.i = feed_forward(
            input_dim=node_embedding_dim * 2,
            num_hidden_layers=2,
            hidden_dim=node_embedding_dim * 2,
            output_dim=node_embedding_dim,
            activation=nn.ReLU,
        )

        self.j = feed_forward(
            input_dim=node_embedding_dim,
            num_hidden_layers=2,
            hidden_dim=node_embedding_dim * 2,
            output_dim=node_embedding_dim,
            activation=nn.ReLU,
        )

        self.soft = nn.Softmax()

        self.output_feed_forward = feed_forward(
            input_dim=node_embedding_dim,
            num_hidden_layers=1,
            hidden_dim=node_embedding_dim * 2,
            output_dim=2,
            activation=nn.ReLU,
        )

    def forward(self, data):
        node_attr, edge_attr, edge_index, batch = (data.x.float(), data.edge_attr.float(), data.edge_index, data.batch)

        node_attr = self.node_embedding(node_attr)
        edge_attr = self.edge_embedding(edge_attr)

        h = node_attr
        for mpnn_layer in self.mpnn_layers:
            h = mpnn_layer(h, edge_index, edge_attr)

        R = self.readout(h, node_attr, batch)

        outputs = self.output_feed_forward(R)

        return F.log_softmax(outputs, dim=-1)

    def readout(self, h, x, batch):

        h_final = torch.cat([h, x], dim=-1)

        i_output = self.i(h_final)
        j_output = self.j(h)

        activation_i = self.soft(i_output)

        mult = activation_i * j_output

        return global_add_pool(mult, batch)
