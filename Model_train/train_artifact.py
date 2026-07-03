import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftFocalLoss(nn.Module):
    def __init__(self, soft_targets, penalty_matrix, weight = None, gamma = 2.0, reduction = 'mean', device = 'cuda'):
        super(SoftFocalLoss,self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
        soft_tensor = torch.tensor(soft_targets, dtype=torch.float32)
        self.register_buffer('soft_targets', soft_tensor.to(device))

        pnl_tensor = torch.tensor(penalty_matrix, dtype = torch.float32)
        self.register_buffer('penalty_matrix', pnl_tensor.to(device))

    def forward(self, logits, labels):
        targets = self.soft_targets[labels]
        ce_loss =  F.cross_entropy(logits, targets, weight = self.weight, reduction = 'none')
        probs = F.softmax(logits, dim = 1)
        
        pt = (probs * targets).sum(dim = 1)
        focal_loss = ((1-pt) ** self.gamma) * ce_loss

        batch_penalty = self.penalty_matrix[labels]
        expected_penalty = (probs * batch_penalty).sum(dim = 1)

        total_loss = focal_loss * (1.0 + expected_penalty)

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
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