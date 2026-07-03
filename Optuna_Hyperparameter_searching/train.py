import logging
import os
import shutil
import optuna
import math 
import torch
import numpy as np
from pathlib import Path
from functools import partial
from datetime import datetime
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch import nn
from dataset import create_dataloader
from FE_DeepLOB import FE_DeepLOB, train_engine, validate_engine
from train_artifact import SoftFocalLoss, EarlyStopping
from process_data import process_data, window_normalize_FE
import config

def setup_logger():
    log_dir = "Logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"Optuna_Campaign_{timestamp}.log")
    
    logger = logging.getLogger("DeepLOB_Quant")
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
    optuna.logging.enable_propagation()  
    optuna.logging.disable_default_handler()
    return logger

def objective(trial, logger):
    window_size = trial.suggest_int('window_size', 100, 500, step = 5)

    gamma = trial.suggest_float("gamma", 1.2, 3)
    # soft matrix
    target_conf = trial.suggest_float('target_conf', 0.85, 0.99)
    target_conf_mid = trial.suggest_float('target_conf_mid', 0.85, 0.99)
    rem = 1.0 - target_conf
    rem_mid = (1 - target_conf_mid) / 2
    soft_matrix = [
        [target_conf, rem, 0.0],
        [rem_mid, target_conf_mid, rem_mid],
        [0.0, rem, target_conf]
    ]

    # PnL matrix
    max_p = trial.suggest_float("max_penalty", 1.0, 3.0)
    mid_p = trial.suggest_float("mid_penalty", 0.1, 2.0)
    min_p = trial.suggest_float("min_penalty", 0.1, 1.0)
    pnl_matrix = [
        [0.0, min_p, max_p],
        [mid_p, 0.0, mid_p],
        [max_p, min_p, 0.0]
    ]

    train_loader, train_label = create_dataloader(
            config.normalized_data_path,
            config.train_start_file,
            config.train_num_files,
            window_size,
            config.target_size,
            config.batch_size,
            config.shuffle_train,
            config.drop_last,
        )
    val_loader, _ = create_dataloader(
            config.normalized_data_path,
            config.val_start_file,
            config.val_num_files,
            window_size,
            config.target_size,
            config.batch_size*2,
            False,
            False,
        )

    train_label['Pctg%'] = round(train_label['Count']/train_label['Count'].sum() * 100, 2)
    logger.debug(f'Down Pctg%: {train_label.iloc[0,-1]}, Stationary Pctg%: {train_label.iloc[1, -1]}, Up Pctg%: {train_label.iloc[2, -1]}')

    num_features = config.num_features
    num_epochs = config.num_epochs  # 200
    learning_rate = config.learning_rate
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    warmup_epochs = config.warmup_epochs

    class_weights = train_label['Count'].sum() / (train_label['Count'] * len(train_label))
    weights = torch.tensor(class_weights.values, dtype=torch.float32)

    loss_history = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FE_DeepLOB_model = FE_DeepLOB(num_features=num_features, num_classes=3)
    FE_DeepLOB_model.to(device)

    optimizer = torch.optim.AdamW(FE_DeepLOB_model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max = total_steps - warmup_steps, eta_min=config.eta_min)
        ],
        milestones = [warmup_steps]
    )
    warmup_criterion = nn.CrossEntropyLoss(weight = weights.to(device))
    soft_criterion = SoftFocalLoss(soft_targets=soft_matrix, penalty_matrix=pnl_matrix, weight = weights.to(device), gamma=gamma, reduction='mean', device = device)

    early_stopping = EarlyStopping(patience = config.early_stop_patience, verbose = False, path = None, monitor_loss = True)
        
    for epoch in range(num_epochs):
        # Dynamic Loss Function
        if epoch < warmup_epochs:
            criterion = warmup_criterion
        else:
            if epoch == warmup_epochs: 
                optimizer.state.clear()
            criterion = soft_criterion
                
        avg_loss, avg_acc = train_engine(FE_DeepLOB_model, train_loader, optimizer, criterion, device, lr_scheduler = lr_scheduler)
        loss_history.append(avg_loss)

        avg_val_loss, val_pr_auc_down, val_pr_auc_up= validate_engine(FE_DeepLOB_model, val_loader, criterion, device)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logger.debug(f"\t [Epoch {epoch:3d}] Loss: {avg_val_loss:.4f} | PR-Down: {val_pr_auc_down:.4f} | PR-Up: {val_pr_auc_up:.4f}")
        trial.report(val_pr_auc_down + val_pr_auc_up, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} is pruned at Epoch {epoch}(Score: {val_pr_auc_down + val_pr_auc_up:.4f})")
            raise optuna.exceptions.TrialPruned()
        if epoch >= warmup_epochs:
            early_stopping(avg_val_loss, FE_DeepLOB_model)

            if early_stopping.early_stop:
                logger.info("Early stopping triggered. Training isolated.")
                break
    return val_pr_auc_down + val_pr_auc_up

def main():
    logger = setup_logger()

    # 清空并重建数据目录，确保每次运行从干净状态开始
    for data_dir in [config.processed_data_path, config.normalized_data_path]:
        dir_path = Path(data_dir)
        if dir_path.exists():
            logger.info(f'Cleaning directory: {data_dir}')
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    target_windows = [10, 20, 50, 100]
    target_alphas = [1.0, 1.5, 2.0]

    best_results_log = {}
    for lw in target_windows:
        # 根据窗口大小动态缩放 alpha 阈值: tau = sqrt(lw / 10)
        tau = math.sqrt(lw / 10)
        scaled_alphas = [round(base * tau, 1) for base in target_alphas]
        logger.info(f'Label Window = {lw}, tau = {tau:.4f}, Scaled Alphas = {scaled_alphas}')

        for alpha in scaled_alphas:
            env_name = f'LW_{lw}_Alpha_{alpha}'
            logger.info(f'Current Test Environment: Label Window = {lw}, Alpha = {alpha} bps')

            inputpath = config.raw_data_path
            outputpath = config.processed_data_path
            path = Path(outputpath)
            path.mkdir(parents=True, exist_ok=True)

            logger.info('Preparing data...')
            raw_files = list(Path(inputpath).glob('*.csv'))

            for raw_file in raw_files:
                outputfile = f'{outputpath}processed_{raw_file.stem[-4:]}.csv'
                process_data(raw_file, outputfile, label_method='l2', label_window = lw, alpha = alpha * 1e-4)

            inputpath_normalized = config.processed_data_path
            outputpath_normalized = config.normalized_data_path 
            path = Path(outputpath_normalized)
            path.mkdir(parents=True, exist_ok=True)
            window_normalize_FE(inputpath_normalized, outputpath_normalized, window_size = 5)

            study = optuna.create_study(direction='maximize', study_name=env_name)
            
            logger.info('Start Hyperparameter searching ...')
            frozen_objective = partial(objective, logger=logger)
            
            study.optimize(frozen_objective, n_trials=50)

            logger.info(f"\n[Done] Env {env_name} Ending Searching. Best Parameters: {study.best_params}")
            best_results_log[env_name] = study.best_value
    logger.info(f'Final Result: \n {best_results_log}')

if __name__ == "__main__":
    main()