import warnings
warnings.filterwarnings('ignore')
from optimizers_niapy_gcn import *
from other_optimizers_gcn import *
from other_optimizers_gan import *
from optimizers_niapy_gan import *

def gcn_none():
    model = GCNLinkPredictor(in_channels=5, hidden_channels=256)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\nNo optimizing")
    epoch, loss, auc_gcn, f1_gcn, ndcg = 0,0,0,0,0
    for _ in range(1, 10):
      loss = GCNtrain(model, optimizer, train_data)
      test_probs = GCNtest(model, test_data)
      auc_gcn, f1_gcn, ndcg = evaluate_model(test_probs, test_data.edge_label)
      epoch=_
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, AUC score: {auc_gcn:.4f}, F1 score: {f1_gcn:.4f}, NDCG: {ndcg:.4f}")

def gan_none():
    generator = Generator(in_channels=5, hidden_channels=256).to(device)
    discriminator = Discriminator(hidden_channels=256).to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    print("\nNo optimizing")
    epoch, d_loss, g_loss, auc_gan, f1_gan, ndcg = 0, 0, 0, 0, 0, 0
    for _ in range(1, 10):
        d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
        test_probs = GANtest(generator, discriminator, test_data)
        auc_gan, f1_gan, ndcg = evaluate_model(test_probs, test_data.edge_label)
        epoch = _
    print(f"Epoch {epoch + 1}, Average Loss: {(d_loss - g_loss):.4f}, AUC score: {auc_gan:.4f}, F1 score: {f1_gan:.4f}, NDCG: {ndcg:.4f}")

def gcn_ga():
    result = run_gcn_ga()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_ga():
    result = run_gan_ga()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_pso():
    result = run_gcn_pso()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_pso():
    result = run_gan_pso()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_sa():
    result = run_gcn_sa()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_sa():
    result = run_gan_sa()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_abc():
    result = run_gcn_abc()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_abc():
    result = run_gan_abc()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_aco():
    result = run_gcn_aco(train_data, test_data)

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_aco():
    result = run_gan_aco(train_data, test_data)

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_hc():
    result = run_gcn_hc()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_hc():
    result = run_gan_hc()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_rs():
    result = run_gcn_ra()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_rs():
    result = run_gan_ra()

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_bo():
    result = run_gcn_bo(train_data, test_data)

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_bo():
    result = run_gan_bo(train_data, test_data)

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_optuna():
    result = run_gcn_optuna(train_data, test_data)

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_optuna():
    result = run_gan_optuna(train_data, test_data)

    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gcn_gs():
    result = run_gcn_gs(train_data, test_data)
    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Num Layers      : {result['best_params']['num_layers']}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

def gan_gs():
    result = run_gan_gs(train_data, test_data)
    print("\nBest Hyperparameters Found:")
    print(f"  Hidden Channels : {result['best_params']['hidden_channels']}")
    print(f"  Learning Rate   : {result['best_params']['lr']:.5f}")
    print(f"  Dropout         : {result['best_params']['dropout']:.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print(f"  AUC             : {result['auc']:.4f}")
    print(f"  Average Loss    : {result['loss']:.4f}")
    print(f"  NDCG            : {result['ndcg']:.4f}")

if __name__ == "__main__":
    gcn_none()
    gan_none()
    gcn_ga()
    gan_ga()
    gcn_pso()
    gan_pso()
    gcn_sa()
    gan_sa()
    gcn_abc()
    gan_abc()
    gcn_aco()
    gan_aco()
    gcn_hc()
    gan_hc()
    gcn_rs()
    gan_rs()
    gcn_bo()
    gan_bo()
    gcn_optuna()
    gan_optuna()
    gcn_gs()
    gan_gs()
