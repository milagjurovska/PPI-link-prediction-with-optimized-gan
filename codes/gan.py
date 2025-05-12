from data_processing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        x = F.leaky_relu(self.lin1(x), 0.2)
        x = self.dropout(x)
        return self.lin2(x)


class Discriminator(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels * 2, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        edge_feats = torch.cat([src, dst], dim=1)
        h = F.leaky_relu(self.lin1(edge_feats), 0.2)
        h = self.dropout(h)
        h = F.leaky_relu(self.lin2(h), 0.2)
        return self.lin3(h)

generator = Generator(in_channels=5, hidden_channels=256).to(device)
discriminator = Discriminator(hidden_channels=256).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))


def GANtrain():
    generator.train()
    discriminator.train()

    z = generator.encode(train_data.x)

    pos_edge_index = train_data.edge_label_index[:, train_data.edge_label == 1]
    neg_edge_index = train_data.edge_label_index[:, train_data.edge_label == 0]

    for _ in range(5):
        optimizer_D.zero_grad()

        pos_pred = discriminator(z.detach(), pos_edge_index)

        neg_pred = discriminator(z.detach(), neg_edge_index)

        d_loss = -(torch.mean(pos_pred) - torch.mean(neg_pred))

        alpha = torch.rand(pos_edge_index.size(1), 1, device=device)
        interpolates = (alpha * z[pos_edge_index[0]] +
                        (1 - alpha) * z[neg_edge_index[0]]).requires_grad_(True)
        disc_interpolates = discriminator(interpolates, pos_edge_index)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        d_loss += 10 * gradient_penalty

        d_loss.backward(retain_graph=True)
        optimizer_D.step()

    optimizer_G.zero_grad()
    neg_pred = discriminator(z, neg_edge_index)
    g_loss = -torch.mean(neg_pred)
    g_loss.backward()
    optimizer_G.step()

    return d_loss.item(), g_loss.item()

@torch.no_grad()
def GANtest(data):
    generator.eval()
    z = generator.encode(data.x)
    scores = discriminator(z, data.edge_label_index)
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

    return auc, f1, optimal_threshold