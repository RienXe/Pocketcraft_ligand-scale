"""
Unified Training Pipeline for Protein Pocket Prediction Models.

This script manages the end-to-end lifecycle of training GNN and PointNet++ models. 
It supports three distinct stages:
1. Self-Supervised Pre-training: Learns spatial representations using masked feature reconstruction.
2. Hyperparameter Optimization: Uses Optuna to find the best architectural and training parameters.
3. Final K-Fold Training: Performs robust cross-validation with the optimal parameters.

Supported Architectures: PointNet++, DGCNN, and PointNet.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import optuna

from data.dataset import ProteinPocketDataset
from models.pointnet2 import PointNet2Segmenter
from models.gnn import GNNModel
from training.trainer import Trainer, save_plots

def run_pretrain(args, device, model_class):
    """
    Executes stage 1 (Pre-training) on the entire training dataset.
    
    Args:
        args: Command-line arguments.
        device: Torch device (CPU/CUDA).
        model_class: The model class to instantiate.
    
    Returns:
        str: Path to the saved pretrained weights.
    """
    print("\n" + "="*50)
    print(">>> STAGE 1: Self-Supervised Masked Pre-training")
    print("="*50)
    
    train_loader = DataLoader(ProteinPocketDataset(args.dir_train, is_train=True), 
                              batch_size=args.batch_size, shuffle=True)
                              
    if args.model_type == "pointnet2":
        model = PointNet2Segmenter(args.in_channels, 1, args.hidden_dim).to(device)
    else:
        model = GNNModel(args.model_type, in_channels=args.in_channels, hidden_dim=args.hidden_dim).to(device)
                               
    trainer = Trainer(model, device, lr=1e-3)
    
    for epoch in range(1, args.pretrain_epochs + 1):
        loss = trainer.pretrain_epoch(train_loader)
        print(f"  Pre-train Epoch {epoch:03d}/{args.pretrain_epochs:03d} | Loss: {loss:.4f}")
    
    pretrained_path = os.path.join(args.log_dir, f"{args.model_type}_pretrained.pth")
    torch.save(model.state_dict(), pretrained_path)
    return pretrained_path

def objective(trial, args, device, pretrained_path, train_files, val_files):
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    if args.model_type == "pointnet2":
        sa1_r = trial.suggest_float("sa1_r", 4.0, 12.0)
        sa2_r = trial.suggest_float("sa2_r", 10.0, 25.0)
        sa1_ratio = trial.suggest_float("sa1_ratio", 0.2, 0.5)
        sa2_ratio = trial.suggest_float("sa2_ratio", 0.1, 0.4)
        fp_k = trial.suggest_int("fp_k", 2, 5)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = PointNet2Segmenter(args.in_channels, 1, args.hidden_dim,
                                   sa1_ratio=sa1_ratio, sa1_r=sa1_r,
                                   sa2_ratio=sa2_ratio, sa2_r=sa2_r,
                                   fp_k=fp_k, dropout=dropout).to(device)
    else:
        k = trial.suggest_int("k", 10, 30)
        model = GNNModel(args.model_type, in_channels=args.in_channels, hidden_dim=args.hidden_dim, k=k).to(device)
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    
    trainer = Trainer(model, device, lr=lr, weight_decay=weight_decay)
    
    train_dataset = ProteinPocketDataset(args.dir_train, files=train_files, is_train=True, voxel_size=args.tuning_voxel_size)
    val_dataset = ProteinPocketDataset(args.dir_train, files=val_files, is_train=False, voxel_size=args.tuning_voxel_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    for epoch in range(args.tuning_epochs):
        trainer.train_epoch(train_loader)
        val_loss, _, _ = trainer.evaluate(val_loader)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline for Protein Pocket Prediction Models using GNN or PointNet architectures.")
    parser.add_argument("--dir_train", required=True, help="Directory containing the training dataset of processed point cloud tensors (.pt).")
    parser.add_argument("--dir_test", required=True, help="Directory containing the testing dataset of processed point cloud tensors (.pt).")
    parser.add_argument("--model_type", choices=["dgcnn", "pointnet", "pointnet2"], default="pointnet2", help="Neural network architecture to train.")
    parser.add_argument("--in_channels", type=int, default=32, help="Number of input feature channels present in the dataset tensors.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensionality of the hidden representation layers in the model.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of point cloud samples per batch during training and evaluation.")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs per fold in the final K-Fold cross-validation stage.")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of splits for K-Fold cross-validation.")
    parser.add_argument("--pretrain_epochs", type=int, default=15, help="Number of epochs for the self-supervised masked pre-training stage.")
    parser.add_argument("--tuning_epochs", type=int, default=15, help="Number of epochs per Optuna trial during hyperparameter tuning.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna hyperparameter optimization trials to run.")
    parser.add_argument("--tuning_voxel_size", type=float, default=None, help="Voxel size for grid downsampling during the hyperparameter tuning stage to speed up execution.")
    parser.add_argument("--log_dir", default="logs_training", help="Directory where model checkpoints, logs, and evaluation plots will be saved.")
    parser.add_argument("--load_pretrained", help="Optional path to load pre-existing model weights, bypassing the self-supervised pre-training stage.")
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = ProteinPocketDataset(args.dir_train, is_train=True)
    all_train_files = full_dataset.files

    pretrained_path = args.load_pretrained
    if not pretrained_path and args.pretrain_epochs > 0:
        pretrained_path = run_pretrain(args, device, None)

    print("\n>>> STAGE 2: Hyperparameter Tuning")
    np.random.seed(42)
    indices = np.random.permutation(len(all_train_files))
    split_val = int(len(all_train_files) * 0.2)
    tuning_val_files = [all_train_files[i] for i in indices[:split_val]]
    tuning_train_files = [all_train_files[i] for i in indices[split_val:]]

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, args=args, device=device, pretrained_path=pretrained_path, 
                           train_files=tuning_train_files, val_files=tuning_val_files), n_trials=args.n_trials)

    print("\nBest Tuning Parameters:", study.best_params)
    best_params = study.best_params

    print("\n>>> STAGE 3: Final K-fold Training")
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    test_loader = DataLoader(ProteinPocketDataset(args.dir_test, is_train=False), batch_size=args.batch_size)
    
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_files)):
        print(f"\n--- Fold {fold+1} ---")
        t_files = [all_train_files[i] for i in train_idx]
        v_files = [all_train_files[i] for i in val_idx]
        
        t_loader = DataLoader(ProteinPocketDataset(args.dir_train, files=t_files, is_train=True), batch_size=args.batch_size, shuffle=True)
        v_loader = DataLoader(ProteinPocketDataset(args.dir_train, files=v_files, is_train=False), batch_size=args.batch_size, shuffle=False)
        
        if args.model_type == "pointnet2":
            model = PointNet2Segmenter(args.in_channels, 1, args.hidden_dim, **{k:v for k,v in best_params.items() if k in ['sa1_ratio', 'sa1_r', 'sa2_ratio', 'sa2_r', 'fp_k', 'dropout']}).to(device)
        else:
            model = GNNModel(args.model_type, in_channels=args.in_channels, hidden_dim=args.hidden_dim, k=best_params.get('k', 20)).to(device)
        
        if pretrained_path:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            
        trainer = Trainer(model, device, lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        
        history = {'train_loss': [], 'val_loss': [], 'val_pr_auc': [], 'val_roc_auc': []}
        best_val_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(t_loader, desc=f"Epoch {epoch}")
            val_loss, pr_auc, roc_auc = trainer.evaluate(v_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_pr_auc'].append(pr_auc)
            history['val_roc_auc'].append(roc_auc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.log_dir, f"best_model_fold_{fold+1}.pth"))
        
        save_plots(history, fold+1, args.log_dir)
        model.load_state_dict(torch.load(os.path.join(args.log_dir, f"best_model_fold_{fold+1}.pth")))
        t_loss, t_pr, t_roc = trainer.evaluate(test_loader)
        fold_metrics.append({'loss': t_loss, 'pr_auc': t_pr, 'roc_auc': t_roc})

    print("\nFinal Results:", pd.DataFrame(fold_metrics).describe().loc[['mean', 'std']])

if __name__ == "__main__":
    main()
