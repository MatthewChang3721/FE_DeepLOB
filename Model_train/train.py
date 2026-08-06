import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import config
from dataset import create_dataloader
from FE_DeepLOB import FE_DeepLOB, train_engine, validate_engine
from process_data import run_pipeline
from train_artifact import EarlyStopping, SoftFocalLoss


def setup_logger():
    os.makedirs(config.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f"Train_{timestamp}.log")

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
    return logger


def clear_directory(dir_path: Path):
    print('---Delete last training files---')
    if dir_path.exists():
        for f in dir_path.iterdir():
            if f.is_file():
                f.unlink()

def prepare_data(logger):
    processed_path = Path(config.processed_data_path)
    normalized_path = Path(config.normalized_data_path)
    clear_directory(processed_path)
    clear_directory(normalized_path)
    processed_path.mkdir(parents=True, exist_ok=True)
    normalized_path.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing data...")
    raw_files = list(Path(config.raw_data_path).glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {config.raw_data_path}")

    run_pipeline(
        label_method=config.label_method,
        label_window=config.label_window,
        alpha=config.alpha,
        normalize_window=config.normalize_window,
        inputpath=config.raw_data_path,
        processedpath=config.processed_data_path,
        outputpath=config.normalized_data_path,
    )
    logger.info("Data preparation finished.")


def train(logger):
    train_loader, train_label = create_dataloader(
        config.normalized_data_path,
        config.train_start_file,
        config.train_num_files,
        config.window_size,
        config.target_size,
        config.batch_size,
        config.shuffle_train,
        config.drop_last,
    )
    val_loader, _ = create_dataloader(
        config.normalized_data_path,
        config.val_start_file,
        config.val_num_files,
        config.window_size,
        config.target_size,
        config.batch_size*2,
        False,
        False,
    )

    train_label["Pctg%"] = round(
        train_label["Count"] / train_label["Count"].sum() * 100, 2
    )
    logger.debug(
        "Down Pctg%%: %s, Stationary Pctg%%: %s, Up Pctg%%: %s",
        train_label.iloc[0, -1],
        train_label.iloc[1, -1],
        train_label.iloc[2, -1],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = FE_DeepLOB(
        num_features=config.num_features,
        num_classes=config.num_classes,
    )
    model.to(device)
    if device.type == 'cuda':
        model = torch.compile(model, mode = 'reduce-overhead')
    else: 
        pass

    class_weights = train_label["Count"].sum() / (
        train_label["Count"] * len(train_label)
    )
    weights = torch.tensor(class_weights.values, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config.eta_min,
            ),
        ],
        milestones=[warmup_steps],
    )

    warmup_criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    soft_criterion = SoftFocalLoss(
        soft_targets=config.soft_matrix,
        penalty_matrix=config.pnl_matrix,
        weight=weights.to(device),
        gamma=config.gamma,
        reduction="mean",
        device=device,
    )

    os.makedirs(config.model_save_dir, exist_ok=True)
    model_path = os.path.join(config.model_save_dir, config.model_save_name)
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        verbose=config.early_stop_verbose,
        path=model_path,
        monitor_loss=config.monitor_loss,
    )

    logger.info("Start training...")
    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        if epoch < config.warmup_epochs:
            criterion = warmup_criterion
        else:
            if epoch == config.warmup_epochs:
                optimizer.state.clear()
            criterion = soft_criterion

        avg_loss, avg_acc = train_engine(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            lr_scheduler=lr_scheduler,
        )

        avg_val_loss, val_pr_auc_down, val_pr_auc_up = validate_engine(
            model, val_loader, criterion, device
        )

        if epoch % config.log_interval == 0 or epoch == config.num_epochs - 1:
            logger.info(
                "[Epoch %3d] Train Loss: %.4f | Val Loss: %.4f | PR-Down: %.4f | PR-Up: %.4f",
                epoch,
                avg_loss,
                avg_val_loss,
                val_pr_auc_down,
                val_pr_auc_up,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if epoch >= config.warmup_epochs:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                logger.info(
                    "Early stopping triggered at epoch %d. Best model saved to %s",
                    epoch,
                    model_path,
                )
                break
    else:
        logger.info("Training finished. Best model saved to %s", model_path)

    date_str = datetime.now().strftime("%Y%m%d")
    alpha_bps = config.alpha * 1e4
    alpha_str = f"{alpha_bps:g}bps"
    final_model_path = os.path.join(
        "Model/torch",
        f"{date_str}_{config.label_window}_{alpha_str}_{config.window_size}.pt",
    )
    os.makedirs("Model/torch", exist_ok=True)
    shutil.copy(model_path, final_model_path)
    logger.info("Best model exported to %s", final_model_path)

    return best_val_loss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(config.seed)
    logger = setup_logger()
    logger.info(
        "Config: label_window=%s, alpha=%s, window_size=%s",
        config.label_window,
        config.alpha,
        config.window_size,
    )

    prepare_data(logger)
    best_val_loss = train(logger)
    logger.info("Training complete. Best validation loss: %.4f", best_val_loss)

if __name__ == "__main__":
    main()
