from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import GeneticAlgorithm, ParticleSwarmOptimization, ArtificialBeeColonyAlgorithm
from niapy.algorithms.other import SimulatedAnnealing, HillClimbAlgorithm, RandomSearch
from gcn import *
from gan import *
from data_processing import train_data, test_data
from niapy.algorithms.algorithm import Individual

class GCNHyperparameterProblem(Problem):
    def __init__(self):
        super().__init__(
            dimension=4,
            lower=[32, 0.0001, 2, 0.0],
            upper=[256, 0.1, 4, 0.7],
            dtype=float
        )

        self.last_f1 = None
        self.last_auc = None
        self.last_loss = None

    def _evaluate(self, x):
        hidden_channels = int(x[0])
        lr = float(x[1])
        num_layers = int(x[2])
        dropout = float(x[3])

        model = GCNLinkPredictor(
            in_channels=5,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_loss = 0
        for epoch in range(10):
            loss = GCNtrain(model, optimizer, train_data)
            total_loss += loss

        test_probs = GCNtest(model, test_data)
        auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

        avg_loss = total_loss / 10

        self.last_f1 = f1
        self.last_auc = auc
        self.last_loss = avg_loss

        return -f1


def run_gcn_ga():
    problem = GCNHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = GeneticAlgorithm(
        population_size=10,
        crossover_rate=0.8,
        mutation_rate=0.2,
        individual_type = Individual
    )

    best_solution, best_score = algo.run(task)

    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "num_layers": int(best_solution[2]),
            "dropout": best_solution[3]
        },
        "f1": -best_score,
        "auc": problem.last_auc,
        "loss": problem.last_loss
    }

def run_gcn_pso():
    problem = GCNHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = ParticleSwarmOptimization(
        population_size=10,
        c1=2.0,
        c2=2.0,
        w=0.7
    )

    best_solution, best_score = algo.run(task)

    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "num_layers": int(best_solution[2]),
            "dropout": best_solution[3]
        },
        "f1": -best_score,
        "auc": problem.last_auc,
        "loss": problem.last_loss
    }

def run_gcn_abc():
    problem = GCNHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = ArtificialBeeColonyAlgorithm(
        population_size=10,
        limit=100
    )

    best_solution, best_score = algo.run(task)

    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "num_layers": int(best_solution[2]),
            "dropout": best_solution[3]
        },
        "f1": -best_score,
        "auc": problem.last_auc,
        "loss": problem.last_loss
    }


def run_gcn_sa():
    problem = GCNHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = SimulatedAnnealing(
        t_min=0.001,
        t_max=1000.0,
        alpha=0.95
    )

    best_solution, best_score = algo.run(task)

    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "num_layers": int(best_solution[2]),
            "dropout": best_solution[3]
        },
        "f1": -best_score,
        "auc": problem.last_auc,
        "loss": problem.last_loss
    }

def run_gcn_hc():
    problem = GCNHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = HillClimbAlgorithm(
        delta=0.1
    )

    best_solution, best_score = algo.run(task)

    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "num_layers": int(best_solution[2]),
            "dropout": best_solution[3]
        },
        "f1": -best_score,
        "auc": problem.last_auc,
        "loss": problem.last_loss
    }

def run_gcn_ra():
    problem = GCNHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = RandomSearch()

    best_solution, best_score = algo.run(task)

    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "num_layers": int(best_solution[2]),
            "dropout": best_solution[3]
        },
        "f1": -best_score,
        "auc": problem.last_auc,
        "loss": problem.last_loss
    }



class GANHyperparameterProblem(Problem):
    def __init__(self):
        super().__init__(
            dimension=3,
            lower=[64, 0.00001, 0.1],
            upper=[512, 0.01, 0.5],
            dtype=float
        )
        self.last_f1 = None
        self.last_auc = None
        self.last_loss = None

    def _evaluate(self, x):
        hidden_channels = int(x[0])
        lr = float(x[1])
        dropout = float(x[2])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        generator = Generator(in_channels=5, hidden_channels=hidden_channels).to(device)
        discriminator = Discriminator(hidden_channels=hidden_channels).to(device)

        generator.dropout = torch.nn.Dropout(dropout)

        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        total_d_loss = 0
        total_g_loss = 0
        for _ in range(5):
            d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
            if d_loss > 1.0 or g_loss > 1.0:
                return float('inf')
            total_d_loss += d_loss
            total_g_loss += g_loss

        test_probs = GANtest(generator, discriminator, test_data)
        auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

        avg_loss = (total_d_loss + total_g_loss) / 20

        self.last_f1 = f1
        self.last_auc = auc
        self.last_loss = avg_loss

        return -f1

def run_gan_ga():
    problem = GANHyperparameterProblem()
    task = Task(problem=problem, max_evals=15)

    algo = GeneticAlgorithm(
        population_size=8,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_evals=1,
        individual_type=Individual
    )

    best_solution, best_score = algo.run(task)

    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "dropout": best_solution[3]
        },
        "f1": -best_solution.fitness,
        "auc": problem.last_auc,
        "loss": problem.last_loss
    }

