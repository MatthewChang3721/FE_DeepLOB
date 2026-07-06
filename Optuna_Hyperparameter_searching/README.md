# Futures High-Frequency Direction Prediction with FE_DeepLOB & Optuna

This project utilizes high-frequency Limit Order Book (LOB) data with a feature-engineered neural network model to predict short-term futures price direction (up/down/neutral). It leverages **Optuna** Bayesian optimization to automatically search for optimal hyperparameters, improving performance on the specified evaluation metric.

---

## Project File Overview

### `config.py` — Global Static Configuration

Stores base parameters that remain fixed during Optuna search. Modify here when you need to quickly adjust the training skeleton:

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| **Data Paths** |||
| `copy_data_path` | `/yourdatabase` | Remote source data path |
| `fetch_file_path` | `"Data/SC_Primary.csv"` | Intermediate fetched file path |
| `raw_data_path` | `"Data/Raw_data/"` | Raw LOB CSV directory |
| `processed_data_path` | `"Data/FE_DeepLOB_data/Processed_data/"` | Temporary feature-engineered data directory |
| `normalized_data_path` | `"Data/FE_DeepLOB_data/Normalized_data/"` | Final normalized training data directory |
| `model_save_dir` | `"Model/cache"` | Model save directory |
| `model_save_name` | `"best_model.pt"` | Best model filename |
| `log_dir` | `"Logs/"` | Log output directory |
| **Data Processing** |||
| `label_method` | `"l2"` | Label generation method |
| `alpha` | `1e-4` | Price movement threshold for label generation |
| `label_window` | `10` | Number of ticks to look ahead for price direction |
| `normalize_window` | `5` | Number of historical days for rolling normalization |
| **DataLoader Split** |||
| `train_start_file` | `0` | Training set starting file index |
| `train_num_files` | `25` | Number of training files |
| `val_start_file` | `25` | Validation set starting file index |
| `val_num_files` | `5` | Number of validation files |
| `batch_size` | `1024` | Training batch size |
| `target_size` | `1` | Prediction horizon (fixed) |
| `shuffle_train` | `True` | Whether to shuffle training data |
| `drop_last` | `True` | Whether to drop the last incomplete batch |
| **Model** |||
| `window_size` | `150` | Historical window length (ticks), dynamically overridden by Optuna |
| `num_features` | `5` | Internal model feature dimension |
| `num_classes` | `3` | Number of classes |
| **Reproducibility** |||
| `seed` | `42` | Random seed |
| **Training** |||
| `num_epochs` | `200` | Maximum epochs per trial |
| `learning_rate` | `0.001` | Initial learning rate |
| `weight_decay` | `5e-3` | Optimizer weight decay |
| `warmup_epochs` | `5` | Number of warmup epochs |
| `warmup_ratio` | `0.05` | Warmup steps as fraction of total steps |
| `eta_min` | `5e-5` | Minimum learning rate for scheduler |
| `log_interval` | `10` | Logging interval (epochs) |
| **Loss Function** |||
| `gamma` | `2.0` | Loss function parameter, dynamically overridden by Optuna |
| `soft_matrix` | 3×3 matrix | Label smoothing matrix, dynamically overridden by Optuna |
| `pnl_matrix` | 3×3 matrix | Penalty matrix, dynamically overridden by Optuna |
| **Early Stopping** |||
| `early_stop_patience` | `10` | Early stopping patience |
| `early_stop_verbose` | `False` | Whether to print early stopping details |
| `monitor_loss` | `True` | Whether to monitor validation loss |

---

### `process_data.py` — Data Preprocessing Pipeline

#### `process_data(inputfile, outputfile, label_method, label_window, alpha)`

Performs feature engineering and label generation on a single day's raw LOB CSV file, saving the processed data to the specified path.

**Feature Engineering**: Contains proprietary intellectual property — specific formulas are not disclosed.

**Label Generation**:
- Uses the `label_method` parameter to select a labeling strategy (three predefined methods available), computing `price_move_label` based on future-window price movement (values: -1 / 0 / 1 representing down/neutral/up respectively). The `alpha` threshold controls classification sensitivity.
- Default method: l2.

Label values: `1` = up, `0` = neutral, `-1` = down

#### `window_normalize_FE(inputpath, outputpath, window_size)`

Performs cross-day rolling normalization on processed CSV files, **strictly ensuring no future data leakage**.

- Sorts files by date (`sorted(glob)`)
- Maintains a deque buffer of size `window_size + 1`
- Uses the first `window_size` days in the buffer to compute feature means and standard deviations
- Normalizes the `window_size+1`-th day's data using historical statistics
- Outputs preprocessed training feature files and labels

#### Standalone Execution

`process_data.py` can also be run as a standalone script:

```bash
python process_data.py
```

This reads raw CSV files from `Data/SC_Raw_data/` with default parameters and executes the full preprocessing pipeline.

---

### `dataset.py` — Time Series Dataset & DataLoader

#### `TimeSeriesDataset(Dataset)`

A PyTorch `Dataset` subclass for temporal sliding-window slicing:

- Separates feature columns and label columns from the tensor
- Generates `(X, Y)` sample pairs via sliding windows, with offset ensuring labels come from the last timestep of the feature window
- Automatically constrains index range to prevent out-of-bounds errors

#### `create_dataloader(inputpath, start_files, num_files, window_size, target_size, batch_size, shuffle, drop_last)`

- Reads normalized CSV files from `inputpath`
- Slices files by sorted index for flexible train/validation splitting
- Converts label values to PyTorch-compatible classification format
- Computes label distribution and returns it (for class weight calculation)
- Returns `(DataLoader, label_distribution_df)`

---

### `FE_DeepLOB.py` — Neural Network Model & Training/Validation Engine

#### `FE_DeepLOB(nn.Module)`

Contains proprietary intellectual property — not disclosed.

#### `train_engine(model, train_loader, optimizer, criterion, device, lr_scheduler)`

Single-epoch training function:

- Retrieves batched data from the DataLoader
- Applies gradient clipping
- Returns `(avg_loss, avg_acc)`

#### `validate_engine(model, val_loader, criterion, device)`

Single-epoch validation function:

- Computes PR-AUC for Down (class 0) and Up (class 2)
- Returns `(avg_loss, pr_auc_down, pr_auc_up)`

> PR-AUC is more suitable than accuracy for imbalanced classification, better capturing model performance on minority classes (up/down).

---

### `train_artifact.py` — Training Utilities

#### `SoftFocalLoss(nn.Module)`

A weighted loss function combining label smoothing with profit-and-loss awareness. Contains proprietary intellectual property — the formula is not disclosed. During Optuna search, it is tuned via `gamma` and two 3×3 matrix parameters.

#### `EarlyStopping`

Validation loss-based early stopping:

- `patience` controlled by `config.early_stop_patience`
- `verbose` controlled by `config.early_stop_verbose`
- Optional `monitor_loss` to monitor validation loss or other metrics

---

### `train.py` — Optuna Search Entry Point

Core control script with the following workflow:

1. **Data directory cleanup**: Automatically deletes and recreates cache directories at startup, ensuring a clean state for each run
2. **Outer environment sweep**: Iterates over `target_windows`, dynamically scaling the alpha threshold for each window (scaling factor computed from window size) to maintain reasonable label distributions across different time scales
3. **Data preparation**: For each environment combination, runs `process_data()` and `window_normalize_FE()` to generate corresponding labeled files
4. **Optuna search**: Creates an independent study for each environment, optimizing the objective function over multiple trials
5. **Logging**: All output is written to log files

All training hyperparameters are read from `config.py` for centralized tuning.

---

## Hyperparameter Search Space

Optuna searches the following hyperparameters in each trial:

| Parameter | Type | Search Range | Description |
|-----------|:----:|:------------:|-------------|
| `window_size` | `int` | Range search | Historical window length for model input (ticks) |
| `gamma` | `float` | Range search | Loss function focusing parameter |
| `target_conf` | `float` | Range search | Label smoothing matrix diagonal confidence parameter |
| `target_conf_mid` | `float` | Range search | Label smoothing matrix middle-class confidence parameter |
| `max_penalty` | `float` | Range search | Penalty matrix parameter |
| `min_penalty` | `float` | Range search | Penalty matrix parameter |
| `mid_penalty` | `float` | Range search | Penalty matrix parameter |

The specific construction logic of each matrix parameter involves proprietary intellectual property and is not disclosed.

---

## How to Configure the Hyperparameter Search Space

### Modifying Search Ranges

Adjust `trial.suggest_*` parameters in the `objective()` function of `train.py`.

### Adding/Removing Search Parameters

To fix a parameter (exclude it from search), assign a constant value directly in `objective()` instead of using `trial.suggest_*`.

To add a new search parameter:

```python
# Example: searching dropout rate
dropout = trial.suggest_float('dropout', 0.1, 0.5)
```

### Adjusting Trials per Environment

Modify the `n_trials` parameter in `study.optimize()`.

---

## Running the Pipeline

Refer to `Run.txt`:

1. Create and activate a Python virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Install GPU-enabled PyTorch (CUDA 12.1 recommended)
4. Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
5. **Standalone data preprocessing** (optional): `python process_data.py`
6. **Run full pipeline**: `python train.py`
7. Check results in the log files

> `train.py` automatically cleans and rebuilds data cache directories at startup, ensuring a clean state for each run.

---

## Directory Structure

```
Optuna_Hyperparameter_searching/
├── config.py               # Global static configuration
├── process_data.py         # Data preprocessing
├── dataset.py              # Time series dataset & DataLoader
├── FE_DeepLOB.py           # Neural network model + training/validation engine
├── train_artifact.py       # Training utilities
├── train.py                # Optuna entry point
├── requirements.txt        # Python dependencies
├── Run.txt                 # Run instructions
├── README.md               # This file (English)
├── README(Chinese).md      # Chinese version
│
├── Data/
│   ├── Raw_data/           # Raw LOB CSV files
│   └── FE_DeepLOB_data/
│       ├── Processed_data/ # Feature-engineered data (temporary)
│       └── Normalized_data/ # Normalized final training data
│
└── Logs/                   # Optuna search logs