# Data paths
copy_data_path = rf"T:\microvast-gtja_984\Future\CFFEX\IM"
fetch_file_path = "Data/Contract.csv"
raw_data_path = "Data/Raw_data/"
processed_data_path = "Data/FE_DeepLOB_data/Processed_data/"
normalized_data_path = "Data/FE_DeepLOB_data/Normalized_data/"
model_save_dir = "Model/cache"
model_save_name = "best_model.pt"
log_dir = "Logs/"

# Data processing
label_method = "trend"  # 'trend' = midprice rolling | 'path' = triple barrier
alpha = 1 * 1e-4  # bps
label_window = 10  # ticks
normalize_window = 5

# Dataloader split
train_start_file = 0
train_num_files = 20
val_start_file = 20
val_num_files = 5
batch_size = 1024
target_size = 1
shuffle_train = True
drop_last = True

# Model (Get from Optuna Hyperparameter Searching)
window_size = 150  # ticks
num_features = 5
num_classes = 3

# Reproducibility
seed = 42

# Training
num_epochs = 200
learning_rate = 0.001
weight_decay = 5e-3
warmup_epochs = 5
warmup_ratio = 0.05
eta_min = 5e-5
log_interval = 10

# Loss
gamma = 2.0
soft_matrix = [
    [0.90, 0.10, 0.00],
    [0.05, 0.90, 0.05],
    [0.00, 0.10, 0.90],
]
pnl_matrix = [
    [0.0, 0.5, 3.0],
    [1.0, 0.0, 1.0],
    [3.0, 0.5, 0.0],
]

# Early stopping
early_stop_patience = 15
early_stop_verbose = False
monitor_loss = True
