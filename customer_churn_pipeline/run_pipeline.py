"""End-to-End Orchestrator for the Customer Churn ML Pipeline."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import split_and_save_data
from src.evaluate import compare_and_select_champion
from src.train import train_all_candidates


def main():
    start_time = time.time()
    print("=" * 75)
    print("CUSTOMER CHURN PREDICTION PIPELINE: TRAINING & EVALUATION")
    print("=" * 75)

    print("\n[Step 1/3] Loading raw data, engineering targets, and splitting...")
    X_train, X_test, y_train, y_test = split_and_save_data()
    print(f"    Train set size: {X_train.shape[0]} samples, {X_train.shape[1]} raw features")
    print(f"    Test set size:  {X_test.shape[0]} samples, {X_test.shape[1]} raw features")
    print(f"    Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")

    print("\n[Step 2/3] Training and hyperparameter tuning candidate models with 5-Fold CV...")
    trained_models = train_all_candidates(X_train, y_train)

    print("\n[Step 3/3] Evaluating on hold-out test set & simulating business impact...")
    champion_name, metadata = compare_and_select_champion(trained_models, X_test, y_test)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"PIPELINE RUN COMPLETED IN {elapsed:.2f} SECONDS")
    print(f"Selected Champion: {champion_name}")
    print(f"Test ROC-AUC:      {metadata['test_roc_auc']:.4f}")
    print(f"Test PR-AUC:       {metadata['test_pr_auc']:.4f}")
    print(f"Optimal Threshold: {metadata['optimal_decision_threshold']:.2f}")
    print(f"Optimal Savings:   ${metadata['business_impact']['optimal_net_savings']:,.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
