from data_processing import *
import torch
from torch_geometric.nn import GCNConv
import torch.optim as optim


def decode(z, edge_label_index):
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=-1)


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


model = GCNLinkPredictor(in_channels=5, hidden_channels=256)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    z = model.encode(train_data.x, train_data.edge_index)
    optimizer.zero_grad()

    edge_index = train_data.edge_label_index
    edge_labels = train_data.edge_label
    pos_mask = edge_labels == 1

    pos_out = decode(z, edge_index[:, pos_mask])
    neg_out = decode(z, edge_index[:, ~pos_mask])

    eps = 1e-15
    pos_loss = -torch.log(torch.sigmoid(pos_out) + eps).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + eps).mean()
    total_loss = pos_loss + neg_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    scores = decode(z, data.edge_label_index)
    probs = torch.sigmoid(scores)
    return probs.cpu().numpy()