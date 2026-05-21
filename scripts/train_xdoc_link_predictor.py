#!/usr/bin/env python3
"""
Phase 2: Train cross-document citation link predictor.

Uses the feature matrix from Phase 1 to train an XGBoost classifier.
Evaluates via document-stratified cross-validation (split on source docs).

Output:
  - data/04_xdoc_citation/model.json (XGBoost model + feature names)
  - data/04_xdoc_citation/training_report.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, confusion_matrix)
import xgboost as xgb
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-file', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'features_train.npz'))
    parser.add_argument('--metadata-file', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'feature_metadata.json'))
    parser.add_argument('--pair-file', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'pair_records.jsonl'))
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation'))
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # Load data
    print("Loading features...")
    data = np.load(args.feature_file)
    X, y = data['X'], data['y']
    print(f"Feature matrix: {X.shape}, labels: {y.shape}")

    with open(args.metadata_file) as f:
        meta = json.load(f)
    feature_names = meta['feature_names']
    print(f"Features: {feature_names}")
    print(f"Pos/Neg: {meta['n_positive']}/{meta['n_negative']}")

    # Load pair records for doc-level grouping
    pairs = []
    with open(args.pair_file) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    source_docs = np.array([p['source_doc'] for p in pairs])
    unique_source_docs = sorted(set(source_docs))
    print(f"Unique source docs: {len(unique_source_docs)}")

    # --- Document-stratified evaluation ---
    # Split source docs into train/test groups
    kf = GroupKFold(n_splits=args.n_folds)
    fold_results = []

    print(f"\n=== {args.n_folds}-fold Group CV (split by source doc) ===\n")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y, groups=source_docs)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle class imbalance
        scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42 + fold_idx,
            n_jobs=4,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= args.threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        # Top-K precision: at K=10, what fraction of top predictions are correct?
        top_k = min(50, len(y_test))
        top_idx = np.argsort(y_prob)[-top_k:]
        top_k_precision = y_test[top_idx].mean()

        result = {
            'fold': fold_idx,
            'n_train': int(len(y_train)),
            'n_test': int(len(y_test)),
            'n_pos_test': int(y_test.sum()),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'auc': round(float(auc), 4),
            'avg_precision': round(float(ap), 4),
            'top_50_precision': round(float(top_k_precision), 4),
        }
        fold_results.append(result)
        print(f"Fold {fold_idx}: AUC={auc:.4f} AP={ap:.4f} F1={f1:.4f} "
              f"Prec={precision:.4f} Rec={recall:.4f} Top50={top_k_precision:.4f}")

    # Summary
    avg = {}
    for metric in ['precision', 'recall', 'f1', 'auc', 'avg_precision', 'top_50_precision']:
        values = [r[metric] for r in fold_results]
        avg[metric] = round(float(np.mean(values)), 4)
        avg[f'{metric}_std'] = round(float(np.std(values)), 4)

    print(f"\n=== Average across folds ===")
    for k, v in avg.items():
        print(f"  {k}: {v}")

    # --- Train final model on all data ---
    print("\nTraining final model on all data...")
    scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=4,
    )
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importance = final_model.feature_importances_
    importance_sorted = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    print("\nFeature importance:")
    for name, imp in importance_sorted:
        print(f"  {name}: {imp:.4f}")

    # --- Save model ---
    model_path = out_dir / 'xgb_link_predictor.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)

    # Also save as JSON-friendly format
    model_info = {
        'feature_names': feature_names,
        'feature_importance': {name: float(imp) for name, imp in importance_sorted},
        'n_estimators': 200,
        'max_depth': 5,
        'threshold': args.threshold,
        'cv_results': {
            'folds': fold_results,
            'average': avg,
        },
        'training_size': int(len(y)),
        'n_positive': int(y.sum()),
    }
    with open(out_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Model info saved to {out_dir / 'model_info.json'}")

    # --- Find optimal threshold ---
    # Use the final model's predictions
    all_probs = final_model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, all_probs)
    # Find threshold that maximizes F1
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_thresh_idx = np.argmax(f1s[:-1])
    optimal_threshold = float(thresholds[best_thresh_idx])
    optimal_f1 = float(f1s[best_thresh_idx])
    print(f"\nOptimal F1 threshold: {optimal_threshold:.4f} (F1={optimal_f1:.4f})")

    with open(out_dir / 'training_report.json', 'w') as f:
        json.dump({
            **model_info,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
        }, f, indent=2)


if __name__ == '__main__':
    main()
