from niapy.algorithms.basic import GeneticAlgorithm, ArtificialBeeColonyAlgorithm, ParticleSwarmAlgorithm
from gcn import *
import torch
import torch.optim as optim

def fitness(solution):
    hidden_channels = int(solution[0])
    lr = float(solution[1])

    model = GCNLinkPredictor(in_channels=5, hidden_channels=hidden_channels)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(3):
        GCNtrain()

    probs = GCNtest(val_data)
    auc, f1, _ = evaluate_model(probs, val_data.edge_label)

    return -f1