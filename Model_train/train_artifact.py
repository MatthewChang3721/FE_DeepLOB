import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftFocalLoss(nn.Module):
    # Proprietary core algorithm redacted for confidentiality.
            return total_loss
        
class EarlyStopping:
    def __init__(self, patience = 20, verbose = False , detla = 0.0, path = 'checkpoint.pth', monitor_loss = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_min = np.inf if monitor_loss else -np.inf
        self.delta = detla
        self.path = path
        self.monitor_loss = monitor_loss

    def __call__(self, val_metric, model):
        score = -val_metric if self.monitor_loss else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: 
                print(f'{val_metric}. Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            metric_name = 'Loss' if self.monitor_loss else 'Metric'
            print(f'Validation {metric_name} improved ({self.val_min:.6f} --> {val_metric}). Saving model ...')
        if self.path is not None:
            torch.save(model._orig_mod.state_dict(), self.path)
        self.val_min = val_metric
