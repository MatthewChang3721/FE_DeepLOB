# FE-DeepLOB

Deep learning framework for futures L2 Limit Order Book price direction prediction. The project consists of two independent modules: hyperparameter search and model training.

---

## Repository Structure

```
FE-DeepLOB/
├── README.md                       # This file (project overview)
│
├── Optuna_Hyperparameter_searching/ # Hyperparameter optimization with Optuna
│   ├── config.py                   # Static configuration
│   ├── process_data.py             # Feature engineering + label generation + normalization
│   ├── dataset.py                  # Sliding window dataset & DataLoader
│   ├── FE_DeepLOB.py               # Model definition + train/validate engine
│   ├── train_artifact.py           # Custom loss function + EarlyStopping
│   ├── train.py                    # Optuna search entry point
│   ├── requirements.txt            # Dependencies
│   ├── README.md                   # Full documentation (English)
│   └── README(Chinese).md          # Full documentation (Chinese)
│
├── Model_train/                    # Production model training
│   ├── config.py                   # Global configuration
│   ├── data_fetch.py               # Fetch raw data from remote server
│   ├── process_data.py             # Feature engineering + label generation + normalization
│   ├── dataset.py                  # Sliding window dataset & DataLoader
│   ├── FE_DeepLOB.py               # Model definition + train/validate engine
│   ├── train_artifact.py           # Custom loss function + EarlyStopping
│   ├── train.py                    # Training pipeline
│   ├── visualization.py            # Signal analysis & visualization
│   ├── model_analysis.py           # Model evaluation & diagnostics
│   ├── export_model.py             # PyTorch → ONNX export
│   ├── requirements.txt            # Dependencies
│   └── README.md                   # Full documentation (English)
│
└── InferenceC++/                   # C++ inference engine (ONNX Runtime)
    └── src/                        # Inference source code
```

## Modules

### 1. Hyperparameter Search (`Optuna_Hyperparameter_searching/`)

Uses **Optuna** Bayesian optimization to automatically search for optimal hyperparameters across multiple time windows. Each trial explores different combinations of model architecture parameters, loss function coefficients, and training configurations.

Key features:
- Automated hyperparameter search over configurable search spaces
- Multi-window environment sweep with adaptive alpha scaling
- Early stopping within each trial to prune unpromising runs
- Comprehensive logging of all search results

### 2. Model Training (`Model_train/`)

Production training pipeline that trains the FE-DeepLOB model with optimized hyperparameters, evaluates signal quality, and exports to ONNX for deployment.

Key features:
- Two-phase training: CrossEntropy warmup → custom PnL-aware loss function
- Sliding window normalization preventing future data leakage
- Confidence-thresholded signal evaluation
- Comprehensive visual diagnostics (signal histograms, PR curves, scatter plots)
- ONNX export for C++ inference

### 3. C++ Inference (`InferenceC++/`)

ONNX Runtime-based inference engine for real-time deployment.

---

## Quick Start

Each module has its own detailed documentation and `requirements.txt`. See the respective `README.md` for setup and usage instructions.

### Hyperparameter Search

```bash
cd Optuna_Hyperparameter_searching
pip install -r requirements.txt
python opt_searching.py
```

### Model Training

```bash
cd Model_train
pip install -r requirements.txt
python data_fetch.py
python train.py
python export_model.py