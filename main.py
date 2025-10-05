#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_utils import load_metadata, load_data, age_based_mixup, get_age_thresholds, shuffle, uniform_mixup
from models.tlm import TLM
from models.baselines import train_all_baselines, evaluate_all_baselines


def main():
    parser = argparse.ArgumentParser(description='TLM: Tree of Linear Models for Voice Aging')
    parser.add_argument('data_folder', type=str, 
                        help='Path to folder containing embeddings (.npy files)')
    parser.add_argument('train_metadata', type=str,
                        help='Path to training metadata file with format: file_path|speaker_id|age')
    parser.add_argument('test_metadata', type=str,
                        help='Path to test metadata file with format: file_path|speaker_id|age')
    parser.add_argument('--oracle', action='store_true',
                        help='Use oracle mode (skip classifier training, use true labels for routing)')
    parser.add_argument('--max_depth', type=int, default=4,
                        help='Maximum depth for TLM tree (default: 4)')

    args = parser.parse_args()
    
    print("="*60)
    print("TLM: Tree of Linear Models for Voice Aging")
    print("="*60)
    print(f"Data folder: {args.data_folder}")
    print(f"Train metadata: {args.train_metadata}")
    print(f"Test metadata: {args.test_metadata}")
    print(f"Oracle mode: {args.oracle}")
    print(f"Max depth: {args.max_depth}")
    print("="*60)
    

    
    # Load metadata
    print("Loading metadata...")
    metadata_train = load_metadata(args.train_metadata)
    metadata_test = load_metadata(args.test_metadata)
    print(f"Training samples: {len(metadata_train)}")
    print(f"Test samples: {len(metadata_test)}")
    
    # Load embeddings and labels
    print("Loading embeddings...")
    X_train, X_test, y_train, y_test = load_data(metadata_train, metadata_test, args.data_folder)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Age range: {y_train.min():.0f} - {y_train.max():.0f}")
    
    # Apply age-based mixup augmentation
    print("\nApplying age-based mixup augmentation...")
    X_train, y_train = shuffle(X_train, y_train)
    X_train_aug, y_train_aug = age_based_mixup(X_train, y_train)
    print(f"After augmentation: {len(X_train_aug)} training samples")
    print(f"Age range after augmentation: {y_train_aug.min():.0f} - {y_train_aug.max():.0f}")
    
    # Get age thresholds for tree splitting
    thresholds = get_age_thresholds(y_train)
    print(f"Age thresholds for splitting: {thresholds}")
    
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    
    # Train baseline models
    baselines = train_all_baselines(X_train_aug, y_train_aug)
    
    # Evaluate baselines
    print("\nEvaluating baseline models...")
    baseline_results = evaluate_all_baselines(baselines, X_test, y_test)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    for model_name, metrics in baseline_results.items():
        print(f"{model_name:20} | MAE: {metrics['MAE']:.3f} | RMSE: {metrics['RMSE']:.3f}")
    
    print("\n" + "="*60)
    print("TRAINING TLM MODELS")
    print("="*60)
    
    # Train TLM without oracle (normal mode)
    print("\nTraining TLM (Normal Mode)...")
    tlm_normal = TLM(X_train_aug, y_train_aug, max_depth=args.max_depth, split="deterministic")
    tlm_normal.train_node(X_train, y_train, X_test, y_test, thresholds, use_oracle=False)
    
    # Train TLM with oracle 
    tlm_oracle = None
    if args.oracle:
        print("\nTraining TLM (Oracle Mode)...")
        tlm_oracle = TLM(X_train_aug, y_train_aug, max_depth=args.max_depth, split="deterministic")
        tlm_oracle.train_node(X_train, y_train, X_test, y_test, thresholds, use_oracle=True)
    
    print("\n" + "="*60)
    print("TLM EVALUATION")
    print("="*60)
    
    # Evaluate TLM models
    results = {}
    
    # TLM Normal - Hard Routing
    print("Evaluating TLM (Normal, Hard Routing)...")
    pred_normal_hard = tlm_normal.predict(X_test, inference_method="hard")
    mae_normal_hard = mean_absolute_error(y_test, pred_normal_hard)
    rmse_normal_hard = np.sqrt(mean_squared_error(y_test, pred_normal_hard))
    results['TLM_Normal_Hard'] = {'MAE': mae_normal_hard, 'RMSE': rmse_normal_hard}
    
    # TLM Normal - Soft Routing
    print("Evaluating TLM (Normal, Soft Routing)...")
    pred_normal_soft = tlm_normal.predict(X_test, inference_method="soft")
    mae_normal_soft = mean_absolute_error(y_test, pred_normal_soft)
    rmse_normal_soft = np.sqrt(mean_squared_error(y_test, pred_normal_soft))
    results['TLM_Normal_Soft'] = {'MAE': mae_normal_soft, 'RMSE': rmse_normal_soft}
    
    # TLM Oracle (if trained)
    if tlm_oracle:
        # Oracle - Hard Routing
        print("Evaluating TLM (Oracle, Hard Routing)...")
        pred_oracle_hard = tlm_oracle.predict(X_test, inference_method="hard", y_true=y_test)
        mae_oracle_hard = mean_absolute_error(y_test, pred_oracle_hard)
        rmse_oracle_hard = np.sqrt(mean_squared_error(y_test, pred_oracle_hard))
        results['TLM_Oracle_Hard'] = {'MAE': mae_oracle_hard, 'RMSE': rmse_oracle_hard}
        
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\nBaseline Models:")
    print("-" * 50)
    for model_name, metrics in baseline_results.items():
        print(f"{model_name:20} | MAE: {metrics['MAE']:.3f} | RMSE: {metrics['RMSE']:.3f}")
    
    print("\nTLM Models:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:20} | MAE: {metrics['MAE']:.3f} | RMSE: {metrics['RMSE']:.3f}")
    


    print("="*60)
if __name__ == "__main__":
    main()