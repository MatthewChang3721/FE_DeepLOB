# FE-DeepLOB Futures Price Direction Prediction

A deep learning project for futures L2 Limit Order Book price movement prediction based on the FE-DeepLOB architecture. The model employs a CNN + Inception + LSTM hybrid architecture with 4 input modalities to perform 3-class classification of future price direction.

---

## Directory Structure

```
.
├── config.py                  # Global configuration
├── train.py                   # Main training script
├── export_model.py            # PyTorch → ONNX export
├── data_fetch.py              # Fetch raw data from remote server
├── process_data.py            # Feature engineering + label generation + window normalization
├── dataset.py                 # Sliding window dataset + DataLoader
├── FE_DeepLOB.py              # Model definition + train/validate engine
├── train_artifact.py          # Custom loss function + EarlyStopping
├── README.md
│
├── Data/
│   ├── Primary.csv             # Trading day / contract mapping table
│   ├── Raw_data/               # Raw L2 market data CSV files
│   └── FE_DeepLOB_data/
│       ├── Processed_data/     # Feature-engineered CSV files
│       └── Normalized_data/    # Window-normalized CSV files
│
├── Model/
│   ├── cache/                  # Best model during training (best_model.pt)
│   ├── torch/                  # Final .pt model files named by hyperparameters
│   └── onnx/                   # ONNX exported models
│
└── Logs/                       # Training logs
```

---

## Core Files

### `config.py` — Central Configuration

#### Data Paths

| Variable | Description |
|---|---|
| `copy_data_path` | Remote raw data source directory |
| `fetch_file_path` | Trading day / contract mapping table |
| `raw_data_path` | Directory for raw CSV market data |
| `processed_data_path` | Output directory for feature-engineered data |
| `normalized_data_path` | Output directory for normalized data |
| `model_save_dir` | Best model cache path during training |
| `model_save_name` | Cache model filename |
| `log_dir` | Training log output directory |

#### Data Processing

| Variable | Description |
|---|---|
| `label_method` | Label generation method; supports multiple strategies |
| `alpha` | Label threshold; price changes exceeding this threshold are labeled as up/down, otherwise stationary |
| `label_window` | Label window (in ticks), configurable |
| `normalize_window` | Normalization sliding window size, configurable |

#### DataLoader Split

| Variable | Description |
|---|---|
| `train_start_file` | Starting file index for training set |
| `train_num_files` | Number of training files, configurable |
| `val_start_file` | Starting file index for validation set |
| `val_num_files` | Number of validation files, configurable |
| `batch_size` | Training batch size |
| `window_size` | Input sequence length (historical ticks), configurable |
| `shuffle_train` | Whether to shuffle training data |
| `drop_last` | Whether to drop the last incomplete batch |

#### Model Architecture (obtained via hyperparameter search)

| Variable | Description |
|---|---|
| `num_features` | Number of original features for the snapshot modality |
| `num_classes` | Number of classes: 0=Down, 1=Stationary, 2=Up |

#### Training Hyperparameters

| Variable | Description |
|---|---|
| `num_epochs` | Total number of training epochs |
| `learning_rate` | Initial learning rate |
| `weight_decay` | AdamW weight decay |
| `warmup_epochs` | Number of warmup epochs |
| `warmup_ratio` | Ratio of warmup steps to total training steps |
| `eta_min` | Minimum learning rate for the scheduler |
| `log_interval` | Logging interval (epochs) |

#### Loss Function

| Variable | Description |
|---|---|
| `gamma` | Focal Loss gamma; controls the weight of hard vs. easy samples |
| `soft_matrix` | Custom soft label smoothing matrix |
| `pnl_matrix` | Custom PnL penalty matrix; asymmetric misclassification weights incorporating transaction costs |

> The loss function combines soft label smoothing with a PnL penalty matrix, enabling the model to account for asymmetric transaction costs during classification. The specific matrix coefficients are core configuration parameters.

#### Early Stopping

| Variable | Description |
|---|---|
| `early_stop_patience` | Stop training if validation loss does not improve for N epochs |
| `early_stop_verbose` | Whether to print early stopping information |
| `monitor_loss` | Monitor validation loss |

---

### `train.py` — Training Pipeline

1. **Logging setup**
2. **Data preparation** — feature engineering + window normalization
3. **DataLoader creation** — train/validation split per configuration
4. **Model initialization** — optional `torch.compile` acceleration
5. **Optimizer & scheduler** — AdamW + learning rate scheduling
6. **Two-phase training**:
   - **Warmup phase**: standard CrossEntropyLoss
   - **Main phase**: switch to custom loss function
7. **Early stopping** — monitor validation loss
8. **Model export**

**Start training**:
```bash
python train.py
```

---

### `FE_DeepLOB.py` — Model Definition

**Model architecture** (forward pass):

```
Input (4 modalities)
    │
    ├── Momentum ──→ Causal Conv Blocks ──→
    ├── Snapshot ──→ Conv + Causal Conv + Inception ──→
    ├── LT Sensor ──→ Dilated Causal Conv ──→
    └── ST Sensor ──→ (direct concatenation)
                           │
                      Concatenate → LSTM
                           │
                      Linear → Logits
```

This file also contains:
- `train_engine()` — training loop
- `validate_engine()` — validation loop, returns loss + Down/Up PR-AUC

---

### `export_model.py` — ONNX Export

Load a trained `.pt` checkpoint into a clean model and export to ONNX format.

```bash
python export_model.py
```

> Ensure model hyperparameters match the training configuration and update the model weight path before exporting.

---

### `dataset.py` — Sliding Window Dataset

- `TimeSeriesDataset`: extracts multiple input modalities from data via a sliding window
- `create_dataloader()`: reads normalized files and returns a `DataLoader`

### `data_fetch.py` — Data Fetching

Copies the required raw market data files from a remote directory to the local `Raw_data/`, based on the trading day and contract code mapping in `Primary.csv`.

```bash
python data_fetch.py
```

### `train_artifact.py` — Training Utilities

- **Custom loss function**: combines soft labels with a PnL penalty matrix
- **`EarlyStopping`**: early stopping mechanism

---

## Usage

### 1. Fetch Data
```bash
python data_fetch.py
```

### 2. Train Model
```bash
python train.py
```

### 3. Export ONNX
```bash
python export_model.py