from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from gcn import *
from gan import *

def run_gcn_gs(train_data, test_data):
    class GCNWrapper(BaseEstimator):
        def __init__(self, hidden_channels=64, lr=0.001, num_layers=2, dropout=0.5):
            self.hidden_channels = hidden_channels
            self.lr = lr
            self.num_layers = num_layers
            self.dropout = dropout
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.avg_loss = None
            self.auc = None
            self.f1 = None

        def fit(self, X=None, y=None):
            self.model = GCNLinkPredictor(
                in_channels=5,
                hidden_channels=int(self.hidden_channels),
                num_layers=int(self.num_layers),
                dropout=self.dropout
            ).to(self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            total_loss = 0
            epochs = 10
            for epoch in range(epochs):
                loss = GCNtrain(self.model, optimizer, train_data)
                total_loss += loss

            self.avg_loss = total_loss / epochs

            test_probs = GCNtest(self.model, test_data).detach().cpu().numpy().flatten()
            y_true = test_data.edge_label.cpu().numpy()

            self.auc = roc_auc_score(y_true, test_probs)
            preds = (test_probs > 0.5).astype(int)
            self.f1 = f1_score(y_true, preds)

            return self

        def predict(self, X=None):
            test_probs = GCNtest(self.model, test_data).detach().cpu().numpy().flatten()
            return (test_probs > 0.5).astype(int)

        def score(self, X=None, y=None):
            return self.f1 if self.f1 is not None else 0

    param_grid = {
        "hidden_channels": [32, 64, 128, 256],
        "lr": [0.0001, 0.001, 0.01, 0.1],
        "num_layers": [2, 3, 4],
        "dropout": [0.0, 0.3, 0.5, 0.7]
    }

    gcn_estimator = GCNWrapper()
    grid_search = GridSearchCV(estimator=gcn_estimator, param_grid=param_grid, scoring='f1', cv=1)
    grid_search.fit(X=[0], y=[0])

    best_wrapper = grid_search.best_estimator_

    return {
        "best_params": {
            "hidden_channels": int(grid_search.best_params_[0]),
            "lr": grid_search.best_params_[1],
            "num_layers": int(grid_search.best_params_[2]),
            "dropout": grid_search.best_params_[3]
        },
        'best_f1': best_wrapper.f1,
        'best_auc': best_wrapper.auc,
        'best_loss': best_wrapper.avg_loss
    }


def run_gan_gs(train_data, test_data):
    class GANWrapper(BaseEstimator):
        def __init__(self, hidden_channels=128, lr=0.001, dropout=0.3):
            self.hidden_channels = hidden_channels
            self.lr = lr
            self.dropout = dropout
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.generator = None
            self.discriminator = None
            self.avg_loss = None
            self.auc = None
            self.f1 = None

        def fit(self, X=None, y=None):
            self.generator = Generator(in_channels=5, hidden_channels=int(self.hidden_channels)).to(self.device)
            self.discriminator = Discriminator(hidden_channels=int(self.hidden_channels)).to(self.device)

            self.generator.dropout = nn.Dropout(self.dropout)

            optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

            total_d_loss = 0
            total_g_loss = 0
            for _ in range(5):
                d_loss, g_loss = GANtrain(self.generator, self.discriminator, optimizer_G, optimizer_D, train_data)
                total_d_loss += d_loss
                total_g_loss += g_loss

            test_probs = GANtest(self.generator, self.discriminator, test_data).detach().cpu().numpy().flatten()
            y_true = test_data.edge_label.cpu().numpy()

            self.auc = roc_auc_score(y_true, test_probs)
            preds = (test_probs > 0.5).astype(int)
            self.f1 = f1_score(y_true, preds)

            self.avg_loss = (total_d_loss + total_g_loss) / 10

            return self

        def predict(self, X=None):
            test_probs = GANtest(self.generator, self.discriminator, test_data).detach().cpu().numpy().flatten()
            return (test_probs > 0.5).astype(int)

        def score(self, X=None, y=None):
            return self.f1 if self.f1 is not None else 0

    param_grid = {
        'hidden_channels': [64, 128, 256],
        'lr': [0.0001, 0.001],
        'dropout': [0.2, 0.4]
    }

    gan_estimator = GANWrapper()
    grid_search = GridSearchCV(estimator=gan_estimator, param_grid=param_grid, scoring='f1', cv=1)
    grid_search.fit(X=[0], y=[0])

    best_wrapper = grid_search.best_estimator_

    return {
        "best_params": {
            "hidden_channels": int(grid_search.best_params_[0]),
            "lr": grid_search.best_params_[1],
            "dropout": grid_search.best_params_[2]
        },
        'best_f1': best_wrapper.f1,
        'best_auc': best_wrapper.auc,
        'best_loss': best_wrapper.avg_loss
    }

def run_gcn_bo(train_data, test_data):
    class GCNWrapper(BaseEstimator):
        def __init__(self, hidden_channels=128, lr=0.01, num_layers=2, dropout=0.3):
            self.hidden_channels = hidden_channels
            self.lr = lr
            self.num_layers = num_layers
            self.dropout = dropout
            self.model = None
            self.optimizer = None
            self.avg_loss = None
            self.auc = None
            self.f1 = None

        def fit(self, X=None, y=None):
            self.model = GCNLinkPredictor(
                in_channels=5,
                hidden_channels=int(self.hidden_channels),
                num_layers=int(self.num_layers),
                dropout=self.dropout
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            total_loss = 0
            for _ in range(10):
                loss = GCNtrain(self.model, self.optimizer, train_data)
                total_loss += loss

            test_probs = GCNtest(self.model, test_data).detach().cpu().numpy().flatten()
            y_true = test_data.edge_label.cpu().numpy()

            self.auc = roc_auc_score(y_true, test_probs)
            preds = (test_probs > 0.5).astype(int)
            self.f1 = f1_score(y_true, preds)

            self.avg_loss = total_loss / 10
            return self

        def predict(self, X=None):
            test_probs = GCNtest(self.model, test_data).detach().cpu().numpy().flatten()
            return (test_probs > 0.5).astype(int)

        def score(self, X=None, y=None):
            return self.f1 if self.f1 is not None else 0

    search_space = {
        'hidden_channels': Integer(32, 256),
        'lr': Real(1e-4, 0.1, prior='log-uniform'),
        'num_layers': Integer(2, 4),
        'dropout': Real(0.0, 0.7)
    }

    gcn_estimator = GCNWrapper()

    opt = BayesSearchCV(
        estimator=gcn_estimator,
        search_spaces=search_space,
        n_iter=20,
        scoring='f1',
        cv=1,
        n_jobs=1,
        verbose=0
    )

    opt.fit(X=[0], y=[0])

    best_model = opt.best_estimator_

    return {
        'best_params': opt.best_params_,
        'best_f1': best_model.f1,
        'best_auc': best_model.auc,
        'best_loss': best_model.avg_loss
    }


def run_gan_bo(train_data, test_data):
    class GANWrapper(BaseEstimator):
        def __init__(self, hidden_channels=128, lr=0.001, dropout=0.3):
            self.hidden_channels = hidden_channels
            self.lr = lr
            self.dropout = dropout
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.generator = None
            self.discriminator = None
            self.avg_loss = None
            self.auc = None
            self.f1 = None

        def fit(self, X=None, y=None):
            self.generator = Generator(in_channels=5, hidden_channels=int(self.hidden_channels)).to(self.device)
            self.discriminator = Discriminator(hidden_channels=int(self.hidden_channels)).to(self.device)

            self.generator.dropout = nn.Dropout(self.dropout)

            optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

            total_d_loss = 0
            total_g_loss = 0
            for _ in range(5):
                d_loss, g_loss = GANtrain(self.generator, self.discriminator, optimizer_G, optimizer_D, train_data)
                total_d_loss += d_loss
                total_g_loss += g_loss

            test_probs = GANtest(self.generator, self.discriminator, test_data).detach().cpu().numpy().flatten()
            y_true = test_data.edge_label.cpu().numpy()

            self.auc = roc_auc_score(y_true, test_probs)
            preds = (test_probs > 0.5).astype(int)
            self.f1 = f1_score(y_true, preds)

            self.avg_loss = (total_d_loss + total_g_loss) / 10

            return self

        def predict(self, X=None):
            test_probs = GANtest(self.generator, self.discriminator, test_data).detach().cpu().numpy().flatten()
            return (test_probs > 0.5).astype(int)

        def score(self, X=None, y=None):
            return self.f1 if self.f1 is not None else 0

    search_space = {
        'hidden_channels': Integer(64, 512),
        'lr': Real(1e-5, 1e-2, prior='log-uniform'),
        'dropout': Real(0.1, 0.5)
    }

    gan_estimator = GANWrapper()

    opt = BayesSearchCV(
        estimator=gan_estimator,
        search_spaces=search_space,
        n_iter=20,
        scoring='f1',
        cv=1,
        n_jobs=1,
        verbose=0
    )

    opt.fit(X=[0], y=[0])

    best_model = opt.best_estimator_

    return {
        'best_params': opt.best_params_,
        'best_f1': best_model.f1,
        'best_auc': best_model.auc,
        'best_loss': best_model.avg_loss
    }

