import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

class Trainer:
    """
    Core training and evaluation logic.
    
    Handles the training loop, self-supervised pre-training, evaluation on 
    validation/test sets, and metric calculation (PR-AUC, ROC-AUC).
    
    Attributes:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run training on.
        optimizer (Optimizer): Torch optimizer.
        criterion (Module): Focal loss function.
    """
    def __init__(self, model, device, lr=1e-3, weight_decay=1e-4, alpha=0.8, gamma=2.0):
        self.model, self.device = model, device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)

    def pretrain_epoch(self, loader):
        self.model.train()
        loss_all = 0
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            # Self-supervised masking logic
            mask = torch.rand(data.x.size(0)) < 0.15
            if not mask.any(): continue
            target = data.x[mask].clone()
            data.x[mask] = 0
            out = self.model(data, pretrain=True)
            loss = F.mse_loss(out[mask], target)
            loss.backward(); self.optimizer.step()
            loss_all += loss.item()
        return loss_all / len(loader) if len(loader) > 0 else 0

    def train_epoch(self, loader, desc="Training"):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=desc, leave=False)
        for data in pbar:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, data.y.float())
            loss.backward(); self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader, desc="Evaluating"):
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0
        for data in tqdm(loader, desc=desc, leave=False):
            data = data.to(self.device)
            logits = self.model(data)
            loss = self.criterion(logits, data.y.float())
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        if not all_preds: return avg_loss, 0.0, 0.0
        y_true, y_score = np.concatenate(all_targets), np.concatenate(all_preds)
        return avg_loss, average_precision_score(y_true, y_score), roc_auc_score(y_true, y_score)

def save_plots(history, fold, output_dir):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_pr_auc'], label='Val PR-AUC')
    plt.title(f'Fold {fold} Metrics')
    plt.xlabel('Epoch'); plt.ylabel('PR-AUC'); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['val_roc_auc'], label='Val ROC-AUC')
    plt.xlabel('Epoch'); plt.ylabel('ROC-AUC'); plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curves_fold{fold}.png"))
    plt.close()
