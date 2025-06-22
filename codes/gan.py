import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, ndcg_score
import numpy as np
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.dropout = nn.Dropout(0.2)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.leaky_relu(x, 0.2)


class Discriminator(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels * 2, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z, edge_index=None, edge_feats=None):
        if edge_feats is None:
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
            edge_feats = torch.cat([src, dst], dim=1)
        h = F.leaky_relu(self.lin1(edge_feats), 0.2)
        h = self.dropout(h)
        h = F.leaky_relu(self.lin2(h), 0.2)
        return self.lin3(h)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

generator = Generator(in_channels=5, hidden_channels=256).to(device)
discriminator = Discriminator(hidden_channels=256).to(device)

generator.apply(init_weights)
discriminator.apply(init_weights)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))


def GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data):
    generator.train()
    discriminator.train()

    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)
    z = generator.encode(x, edge_index)

    edge_label_index = train_data.edge_label_index.to(device)
    edge_label = train_data.edge_label.to(device)

    for i in range(2):
        BATCH_SIZE = 1024

        pos_idx = torch.where(edge_label == 1)[0]
        neg_idx = torch.where(edge_label == 0)[0]

        pos_sample = pos_idx[torch.randperm(len(pos_idx))[:BATCH_SIZE]]
        neg_sample = neg_idx[torch.randperm(len(neg_idx))[:BATCH_SIZE]]

        pos_edge_index = edge_label_index[:, pos_sample]
        neg_edge_index = edge_label_index[:, neg_sample]

        z = generator.encode(x, edge_index)

        optimizer_D.zero_grad()
        pos_pred = discriminator(z, edge_index=pos_edge_index)
        neg_pred = discriminator(z, edge_index=neg_edge_index)
        d_loss = -(pos_pred.mean() - neg_pred.mean())

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        z = generator.encode(x, edge_index)
        neg_pred = discriminator(z, edge_index=neg_edge_index)
        g_loss = -neg_pred.mean()
        g_loss.backward()
        optimizer_G.step()

    optimizer_G.zero_grad()
    neg_pred = discriminator(z, edge_index=neg_edge_index)
    g_loss = -torch.mean(neg_pred)
    optimizer_G.step()

    return d_loss.item(), g_loss.item()

@torch.no_grad()
def GANtest(generator, discriminator, test_data):
    generator.eval()
    discriminator.eval()
    x = test_data.x.to(device)
    edge_index = test_data.edge_index.to(device)
    edge_label_index = test_data.edge_label_index.to(device)
    z = generator.encode(x, edge_index)
    scores = discriminator(z, edge_index=edge_label_index)
    return torch.sigmoid(scores).cpu().numpy()

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

    test_labels = np.asarray(test_labels).reshape(-1)
    test_probs = np.asarray(test_probs).reshape(-1)
    ndcg = ndcg_score([test_labels], [test_probs])

    return auc, f1, ndcg