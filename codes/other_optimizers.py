from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import BaseEstimator
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from itertools import product
import optuna
from gcn import *
from gan import *

def run_gcn_gs(train_data, test_data):
    param_grid = {
        "hidden_channels": list(range(32, 256 + 1, 32)),
        "lr": np.logspace(-4, -1, 5).tolist(),
        "num_layers": [2, 3, 4],
        "dropout": np.round(np.linspace(0.0, 0.7, 4), 2).tolist()
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    grid = list(product(*values))

    best_score = 0
    best_params = None
    best_auc = None
    best_loss = None

    for combination in grid:
        params = dict(zip(keys, combination))

        try:
            model = GCNLinkPredictor(
                in_channels=5,
                hidden_channels=params["hidden_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

            total_loss = 0
            for _ in range(10):
                loss = GCNtrain(model, optimizer, train_data)
                total_loss += loss

            avg_loss = total_loss / 10

            test_probs = GCNtest(model, test_data)
            auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

            if f1 > best_score:
                best_score = f1
                best_params = params
                best_auc = auc
                best_loss = avg_loss

        except Exception as e:
            continue

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "num_layers": int(best_params["num_layers"]),
            "dropout": best_params["dropout"]
        },
        "f1": best_score,
        "auc": best_auc,
        "loss": best_loss
    }

def run_gan_gs(train_data, test_data):
    param_grid = {
        "hidden_channels": list(range(64, 513, 64)),
        "lr": np.logspace(-5, -1, 5).tolist(),
        "dropout": np.round(np.linspace(0.0, 0.7, 4), 2).tolist()
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    grid = list(product(*values))

    best_score = 0
    best_params = None
    best_auc = None
    best_loss = None

    for combination in grid:
        params = dict(zip(keys, combination))

        try:
            generator = Generator(in_channels=5, hidden_channels=params["hidden_channels"]).to(device)
            discriminator = Discriminator(hidden_channels=params["hidden_channels"]).to(device)

            generator.apply(init_weights)
            discriminator.apply(init_weights)

            optimizer_G = torch.optim.Adam(generator.parameters(), lr=params["lr"], betas=(0.5, 0.999))
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=params["lr"], betas=(0.5, 0.999))

            total_d_loss = 0
            total_g_loss = 0
            for _ in range(10):
                d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
                total_d_loss += d_loss
                total_g_loss += g_loss

            avg_loss = (total_d_loss + total_g_loss) / 20

            test_probs = GANtest(generator, discriminator, test_data)
            auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

            if f1 > best_score:
                best_score = f1
                best_params = params
                best_auc = auc
                best_loss = avg_loss

        except Exception as e:
            continue

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "dropout": best_params["dropout"]
        },
        "f1": best_score,
        "auc": best_auc,
        "loss": best_loss
    }

def run_gcn_bo(train_data, test_data, n_init=5, n_iter=15):
    search_space = {
        "hidden_channels": list(range(32, 257, 16)),
        "lr": np.logspace(-4, -1, 20),
        "num_layers": [2, 3, 4],
        "dropout": np.linspace(0.0, 0.7, 15),
    }
    keys = list(search_space.keys())
    space_sizes = [len(search_space[k]) for k in keys]

    def idxs_to_params(idxs):
        return {k: search_space[k][i] for k, i in zip(keys, idxs)}

    X_samples = []
    y_samples = []

    def evaluate_params(params):
        model = GCNLinkPredictor(
            in_channels=5,
            hidden_channels=int(params["hidden_channels"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))

        total_loss = 0
        for _ in range(10):
            loss = GCNtrain(model, optimizer, train_data)
            total_loss += loss.item() if hasattr(loss, 'item') else loss
        avg_loss = total_loss / 10

        test_probs = GCNtest(model, test_data)
        auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)
        return f1, auc, avg_loss

    for _ in range(n_init):
        sample = [random.randint(0, size - 1) for size in space_sizes]
        params = idxs_to_params(sample)
        f1, auc, loss = evaluate_params(params)
        X_samples.append(sample)
        y_samples.append(f1)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)

    gp = GaussianProcessRegressor(kernel=Matern(), normalize_y=True)

    for _ in range(n_iter):
        gp.fit(X_samples, y_samples)

        candidates = np.array([
            [random.randint(0, size - 1) for size in space_sizes]
            for _ in range(100)
        ])

        preds, stds = gp.predict(candidates, return_std=True)
        acquisition = preds + 0.1 * stds

        best_idx = np.argmax(acquisition)
        best_candidate = candidates[best_idx]

        params = idxs_to_params(best_candidate)
        f1, auc, avg_loss = evaluate_params(params)

        X_samples = np.vstack([X_samples, best_candidate])
        y_samples = np.append(y_samples, f1)

    best_idx = np.argmax(y_samples)
    best_indices = X_samples[best_idx]
    best_params = idxs_to_params(best_indices)
    best_f1, best_auc, best_loss = evaluate_params(best_params)

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "num_layers": int(best_params["num_layers"]),
            "dropout": best_params["dropout"],
        },
        'f1':best_f1,
        'auc': best_auc,
        'loss': best_loss
    }


def run_gan_bo(train_data, test_data, n_init=5, n_iter=15):
    search_space = {
        "hidden_channels": list(range(32, 257, 16)),
        "lr": np.logspace(-4, -1, 20),
        "dropout": np.linspace(0.0, 0.7, 15),
    }

    keys = list(search_space.keys())
    space_sizes = [len(search_space[k]) for k in keys]

    def idxs_to_params(idxs):
        return {k: search_space[k][i] for k, i in zip(keys, idxs)}

    X_samples = []
    y_samples = []

    def evaluate_params(params):
        hidden_channels = int(params["hidden_channels"])
        lr = float(params["lr"])
        dropout = float(params["dropout"])

        generator = Generator(in_channels=5, hidden_channels=hidden_channels).to(device)
        discriminator = Discriminator(hidden_channels=hidden_channels).to(device)

        generator.apply(init_weights)
        discriminator.apply(init_weights)

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        total_d_loss, total_g_loss = 0, 0
        for _ in range(10):
            d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
            total_d_loss += d_loss
            total_g_loss += g_loss
        avg_loss = (total_d_loss + total_g_loss) / 20

        test_probs = GANtest(generator, discriminator, test_data)
        auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)
        return f1, auc, avg_loss

    for _ in range(n_init):
        sample = [random.randint(0, size - 1) for size in space_sizes]
        params = idxs_to_params(sample)
        f1, auc, loss = evaluate_params(params)
        X_samples.append(sample)
        y_samples.append(f1)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)

    gp = GaussianProcessRegressor(kernel=Matern(), normalize_y=True)

    for _ in range(n_iter):
        gp.fit(X_samples, y_samples)

        candidates = np.array([
            [random.randint(0, size - 1) for size in space_sizes]
            for _ in range(100)
        ])

        preds, stds = gp.predict(candidates, return_std=True)
        acquisition = preds + 0.1 * stds

        best_idx = np.argmax(acquisition)
        best_candidate = candidates[best_idx]

        params = idxs_to_params(best_candidate)
        f1, auc, loss = evaluate_params(params)

        X_samples = np.vstack([X_samples, best_candidate])
        y_samples = np.append(y_samples, f1)

    best_idx = np.argmax(y_samples)
    best_indices = X_samples[best_idx]
    best_params = idxs_to_params(best_indices)
    best_f1, best_auc, best_loss = evaluate_params(best_params)

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "dropout": best_params["dropout"],
        },
        'f1': best_f1,
        'auc': best_auc,
        'loss': best_loss
    }

def objective(trial):
    hidden_channels = trial.suggest_int("hidden_channels", 32, 256)
    lr = trial.suggest_loguniform("lr", 0.0001, 0.1)
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.7)

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

    avg_loss = total_loss / 10

    test_probs = GCNtest(model, test_data)
    auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

    trial.set_user_attr("f1", f1)
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("loss", avg_loss)

    return -f1

def run_gcn_optuna(n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial

    return {
        "best_params": {
            "hidden_channels": int(best_trial.params["hidden_channels"]),
            "lr": best_trial.params["lr"],
            "num_layers": int(best_trial.params["num_layers"]),
            "dropout": best_trial.params["dropout"]
        },
    "f1": best_trial.user_attrs["f1"],
    "auc": best_trial.user_attrs["auc"],
    "loss": best_trial.user_attrs["loss"]
    }

def objective_gan(trial):
    hidden_channels = trial.suggest_int("hidden_channels", 64, 512)
    lr = trial.suggest_loguniform("lr", 1e-5, 0.1)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.7)

    generator = Generator(in_channels=5, hidden_channels=hidden_channels).to(device)
    discriminator = Discriminator(hidden_channels=hidden_channels).to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    total_d_loss = 0
    total_g_loss = 0
    for epoch in range(10):
        d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
        total_d_loss += d_loss
        total_g_loss += g_loss

    avg_loss = (total_d_loss + total_g_loss) / 20

    test_probs = GANtest(generator, discriminator, test_data)
    auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

    trial.set_user_attr("f1", f1)
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("loss", avg_loss)

    return -f1

def run_gan_optuna(n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_gan, n_trials=n_trials)

    best_trial = study.best_trial

    return {
        "best_params": {
            "hidden_channels": int(best_trial.params["hidden_channels"]),
            "lr": best_trial.params["lr"],
            "dropout": best_trial.params["dropout"]
        },
        "f1": best_trial.user_attrs["f1"],
        "auc": best_trial.user_attrs["auc"],
        "loss": best_trial.user_attrs["loss"]
    }

def run_gcn_aco(train_data, test_data, n_ants=5, n_gen=5, alpha=1, beta=2, evap=0.5):
    search_space = {
        "hidden_channels": list(range(32, 256, 32)),
        "lr": np.logspace(-4, -1, 10).tolist(),
        "num_layers": [2, 3, 4],
        "dropout": np.round(np.linspace(0.0, 0.7, 8), 2).tolist()
    }

    keys = list(search_space.keys())
    values = list(search_space.values())
    n_params = len(keys)
    pheromones = [np.ones(len(v), dtype=np.float64) for v in values]

    best_idx = None
    best_score = 0
    best_auc = None
    best_loss = None

    for _ in range(n_gen):
        solutions = np.zeros((n_ants, n_params), dtype=int)

        for i in range(n_params):
            probs = pheromones[i] ** alpha
            probs /= probs.sum()
            solutions[:, i] = np.random.choice(len(probs), size=n_ants, p=probs)

        scores = np.zeros(n_ants)
        aucs = np.zeros(n_ants)
        losses = np.zeros(n_ants)

        for ant in range(n_ants):
            idxs = solutions[ant]
            try:
                params = {keys[i]: values[i][idxs[i]] for i in range(n_params)}

                model = GCNLinkPredictor(
                    in_channels=5,
                    hidden_channels=params["hidden_channels"],
                    num_layers=params["num_layers"],
                    dropout=params["dropout"]
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

                total_loss = 0
                for epoch in range(10):
                    loss = GCNtrain(model, optimizer, train_data)
                    total_loss += loss
                avg_loss = total_loss / 10

                test_probs = GCNtest(model, test_data)
                auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

                scores[ant] = f1
                aucs[ant] = auc
                losses[ant] = avg_loss

            except Exception as e:
                scores[ant] = 0
                aucs[ant] = 0
                losses[ant] = float("inf")

        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_idx = solutions[max_idx]
            best_auc = aucs[max_idx]
            best_loss = losses[max_idx]

        for i in range(n_params):
            pheromones[i] *= (1 - evap)
            update = np.bincount(solutions[:, i], weights=scores, minlength=len(pheromones[i]))
            pheromones[i][:len(update)] += update

    best_params = {keys[i]: values[i][best_idx[i]] for i in range(n_params)}

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "num_layers": int(best_params["num_layers"]),
            "dropout": best_params["dropout"]
        },
        "f1": best_score,
        "auc": best_auc,
        "loss": best_loss
    }

def run_gan_aco(train_data, test_data, n_ants=5, n_gen=5, alpha=1, beta=2, evap=0.5):
    search_space = {
        "hidden_channels": list(range(64, 513, 64)),
        "lr": np.logspace(-5, -1, 10).tolist(),
        "dropout": np.round(np.linspace(0.0, 0.7, 8), 2).tolist()
    }

    keys = list(search_space.keys())
    values = list(search_space.values())
    n_params = len(keys)
    pheromones = [np.ones(len(v), dtype=np.float64) for v in values]

    best_idx = None
    best_score = 0
    best_auc = None
    best_loss = None

    for _ in range(n_gen):
        solutions = np.zeros((n_ants, n_params), dtype=int)

        for i in range(n_params):
            probs = pheromones[i] ** alpha
            probs /= probs.sum()
            solutions[:, i] = np.random.choice(len(probs), size=n_ants, p=probs)

        scores = np.zeros(n_ants)
        aucs = np.zeros(n_ants)
        losses = np.zeros(n_ants)

        for ant in range(n_ants):
            idxs = solutions[ant]
            try:
                params = {keys[i]: values[i][idxs[i]] for i in range(n_params)}

                generator = Generator(in_channels=5, hidden_channels=params["hidden_channels"]).to(device)
                discriminator = Discriminator(hidden_channels=params["hidden_channels"]).to(device)

                generator.apply(init_weights)
                discriminator.apply(init_weights)

                optimizer_G = torch.optim.Adam(generator.parameters(), lr=params["lr"], betas=(0.5, 0.999))
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=params["lr"], betas=(0.5, 0.999))

                total_d_loss = 0
                total_g_loss = 0
                for _ in range(10):
                    d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
                    total_d_loss += d_loss
                    total_g_loss += g_loss

                avg_loss = (total_d_loss + total_g_loss) / 20

                test_probs = GANtest(generator, discriminator, test_data)
                auc, f1, _ = evaluate_model(test_probs, test_data.edge_label)

                scores[ant] = f1
                aucs[ant] = auc
                losses[ant] = avg_loss

            except Exception as e:
                scores[ant] = 0
                aucs[ant] = 0
                losses[ant] = float("inf")

        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_idx = solutions[max_idx]
            best_auc = aucs[max_idx]
            best_loss = losses[max_idx]

        for i in range(n_params):
            pheromones[i] *= (1 - evap)
            update = np.bincount(solutions[:, i], weights=scores, minlength=len(pheromones[i]))
            pheromones[i][:len(update)] += update

    best_params = {keys[i]: values[i][best_idx[i]] for i in range(n_params)}

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "dropout": best_params["dropout"]
        },
        "f1": best_score,
        "auc": best_auc,
        "loss": best_loss
    }

