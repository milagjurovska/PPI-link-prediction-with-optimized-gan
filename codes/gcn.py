from sklearn.metrics import f1_score, roc_auc_score
from data_processing import *
import torch
from torch_geometric.nn import GCNConv
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

search_space = {
    "hidden_channels": [32, 64, 128, 256],
    "lr": [0.0001, 0.001, 0.01, 0.1],
    "num_layers": [2, 3, 4],
    "dropout": [0.0, 0.3, 0.5, 0.7]
}


def decode(z, edge_label_index):
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=-1)


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def encode(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, edge_index)


model = GCNLinkPredictor(in_channels=5, hidden_channels=256)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def GCNtrain(model, optimizer, train_data):
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
def GCNtest(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    scores = decode(z, data.edge_label_index)
    probs = torch.sigmoid(scores)
    return probs.cpu().numpy()

def find_optimal_threshold(labels, probs):
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def evaluate_model(test_probs, test_labels):
    if torch.is_tensor(test_probs):
        test_probs = test_probs.cpu().numpy()
    if torch.is_tensor(test_labels):
        test_labels = test_labels.cpu().numpy()

    auc = roc_auc_score(test_labels, test_probs)

    optimal_threshold = find_optimal_threshold(test_labels, test_probs)
    preds = (test_probs > optimal_threshold).astype(int)
    f1 = f1_score(test_labels, preds)

    return auc, f1, optimal_threshold