"""Model evaluation, comparative performance analysis, business impact optimization, and visualization."""

import json
import os
from typing import Any, Dict, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    ARTIFACTS_DIR,
    CUSTOMER_LIFETIME_VALUE,
    METADATA_PATH,
    MODEL_PATH,
    RETENTION_OFFER_COST,
    RETENTION_SUCCESS_RATE,
)


def evaluate_predictions(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics for given probabilities and threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "pr_auc": float(average_precision_score(y_true, y_probs)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_probs)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def analyze_business_impact(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    clv: float = CUSTOMER_LIFETIME_VALUE,
    cost: float = RETENTION_OFFER_COST,
    retention_rate: float = RETENTION_SUCCESS_RATE,
) -> Dict[str, Any]:
    """Simulate business value creation across decision thresholds to determine the financially optimal threshold."""
    thresholds = np.linspace(0.05, 0.95, 91)
    results = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        targeted_customers = tp + fp
        campaign_cost = targeted_customers * cost
        retained_customers = tp * retention_rate
        saved_revenue = retained_customers * clv
        net_value_saved = saved_revenue - campaign_cost
        roi_pct = (
            (net_value_saved / campaign_cost * 100.0) if campaign_cost > 0 else 0.0
        )

        results.append(
            {
                "threshold": float(t),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "targeted_count": int(targeted_customers),
                "campaign_cost": float(campaign_cost),
                "retained_customers": float(retained_customers),
                "saved_revenue": float(saved_revenue),
                "net_value_saved": float(net_value_saved),
                "roi_percent": float(roi_pct),
            }
        )

    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results["net_value_saved"].idxmax()]
    default_row = df_results.iloc[(df_results["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

    return {
        "optimal_threshold": float(best_row["threshold"]),
        "optimal_net_savings": float(best_row["net_value_saved"]),
        "optimal_targeted_customers": int(best_row["targeted_count"]),
        "optimal_retained_customers": float(best_row["retained_customers"]),
        "optimal_campaign_cost": float(best_row["campaign_cost"]),
        "optimal_roi_percent": float(best_row["roi_percent"]),
        "default_threshold": float(default_row["threshold"]),
        "default_net_savings": float(default_row["net_value_saved"]),
        "default_targeted_customers": int(default_row["targeted_count"]),
        "default_retained_customers": float(default_row["retained_customers"]),
        "savings_lift_over_default": float(
            best_row["net_value_saved"] - default_row["net_value_saved"]
        ),
        "threshold_curve": results,
    }


def generate_evaluation_visualizations(
    model_evaluations: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    champion_name: str,
    output_dir: str = ARTIFACTS_DIR,
):
    """Plot and save ROC Curves, PR Curves, Confusion Matrices, and Business curves."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, data in model_evaluations.items():
        y_probs = data["y_probs"]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc = data["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Comparison")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=250)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    baseline = np.mean(y_test)
    for name, data in model_evaluations.items():
        y_probs = data["y_probs"]
        prec, rec, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = data["metrics"]["pr_auc"]
        plt.plot(rec, prec, lw=2.2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    plt.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        label=f"Baseline Churn Rate ({baseline:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve Comparison")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=250)
    plt.close()

    # 3. Champion Model Confusion Matrices (Default vs Business-Optimal)
    champ_data = model_evaluations[champion_name]
    y_probs = champ_data["y_probs"]
    opt_thresh = champ_data["business"]["optimal_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Default 0.50 threshold
    cm_default = confusion_matrix(y_test, (y_probs >= 0.5).astype(int))
    sns.heatmap(
        cm_default,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
    )
    axes[0].set_title(f"Default Cutoff (0.50)\nRecall: {cm_default[1,1]/(cm_default[1,1]+cm_default[1,0]):.1%}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("Actual Label")

    # Business Optimal threshold
    cm_opt = confusion_matrix(y_test, (y_probs >= opt_thresh).astype(int))
    sns.heatmap(
        cm_opt,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        ax=axes[1],
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
    )
    axes[1].set_title(
        f"Business-Optimal Cutoff ({opt_thresh:.2f})\nRecall: {cm_opt[1,1]/(cm_opt[1,1]+cm_opt[1,0]):.1%}"
    )
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("Actual Label")

    plt.suptitle(f"{champion_name}: Confusion Matrix Shift", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=250)
    plt.close()

    # 4. Business Value Curve
    plt.figure(figsize=(8, 5))
    thresh_df = pd.DataFrame(champ_data["business"]["threshold_curve"])
    plt.plot(
        thresh_df["threshold"],
        thresh_df["net_value_saved"],
        color="#2b5c8f",
        lw=2.5,
        label="Net Value Saved ($)",
    )
    plt.axvline(
        x=opt_thresh,
        color="#27ae60",
        linestyle="--",
        label=f"Optimal Threshold ({opt_thresh:.2f}): ${champ_data['business']['optimal_net_savings']:,.0f}",
    )
    plt.axvline(
        x=0.50,
        color="#e74c3c",
        linestyle=":",
        label=f"Default Threshold (0.50): ${champ_data['business']['default_net_savings']:,.0f}",
    )
    plt.xlabel("Churn Probability Decision Threshold")
    plt.ylabel("Net Business Value Saved ($)")
    plt.title(f"Expected Net Business Savings vs Decision Threshold ({champion_name})")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "business_curve.png"), dpi=250)
    plt.close()

    # 5. Feature Importances / Coefficients for champion model
    try:
        pipeline = champ_data["pipeline"]
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Get feature names from preprocessor
        feature_names = []
        # numerical
        feature_names.extend(preprocessor.transformers_[0][2])
        # categorical one-hot
        cat_encoder = preprocessor.transformers_[1][1]
        cat_cols = preprocessor.transformers_[1][2]
        cat_ohe_names = cat_encoder.get_feature_names_out(cat_cols)
        feature_names.extend(cat_ohe_names)
        # binary
        feature_names.extend(preprocessor.transformers_[2][2])

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            label = "Feature Importance"
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
            label = "Absolute Logistic Coefficient"
        else:
            importances = None

        if importances is not None and len(importances) == len(feature_names):
            fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            fi_df = fi_df.sort_values(by="Importance", ascending=False).head(15)

            plt.figure(figsize=(9, 6))
            sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
            plt.title(f"Top 15 Most Predictive Features ({champion_name})")
            plt.xlabel(label)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=250)
            plt.close()
    except Exception as e:
        print(f"[!] Warning: Could not plot feature importance: {e}")


def compare_and_select_champion(
    trained_models: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, Dict[str, Any]]:
    """Evaluate all candidates on hold-out test set, perform business evaluation, and select champion."""
    evaluations = {}

    print("\n" + "=" * 70)
    print("HOLD-OUT TEST EVALUATION & BUSINESS IMPACT ANALYSIS")
    print("=" * 70)

    for name, item in trained_models.items():
        pipeline = item["pipeline"]
        y_probs = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics at default 0.5 threshold
        metrics_default = evaluate_predictions(y_test.values, y_probs, threshold=0.5)

        # Calculate business value curve and optimal threshold
        business_impact = analyze_business_impact(y_test.values, y_probs)

        # Calculate metrics at optimal threshold
        opt_thresh = business_impact["optimal_threshold"]
        metrics_optimal = evaluate_predictions(y_test.values, y_probs, threshold=opt_thresh)

        evaluations[name] = {
            "pipeline": pipeline,
            "y_probs": y_probs,
            "cv_roc_auc": item["summary"]["best_cv_roc_auc"],
            "best_params": item["summary"]["best_params"],
            "metrics": metrics_default,
            "metrics_at_optimal_threshold": metrics_optimal,
            "business": business_impact,
        }

        print(f"\nModel: {name}")
        print(f"  CV ROC-AUC:            {item['summary']['best_cv_roc_auc']:.4f}")
        print(f"  Test ROC-AUC:          {metrics_default['roc_auc']:.4f}")
        print(f"  Test PR-AUC:           {metrics_default['pr_auc']:.4f}")
        print(f"  Default F1 (t=0.50):   {metrics_default['f1']:.4f} (Recall: {metrics_default['recall']:.4f}, Prec: {metrics_default['precision']:.4f})")
        print(f"  Optimal Threshold:     {opt_thresh:.2f}")
        print(f"  Optimal F1:            {metrics_optimal['f1']:.4f} (Recall: {metrics_optimal['recall']:.4f}, Prec: {metrics_optimal['precision']:.4f})")
        print(f"  Default Net Savings:   ${business_impact['default_net_savings']:,.2f}")
        print(f"  Optimal Net Savings:   ${business_impact['optimal_net_savings']:,.2f}")
        print(f"  Net Financial Lift:    +${business_impact['savings_lift_over_default']:,.2f}")

    # Select champion based on highest business impact (Optimal Net Savings) and ROC-AUC
    champion_name = max(
        evaluations.keys(),
        key=lambda k: (
            evaluations[k]["business"]["optimal_net_savings"],
            evaluations[k]["metrics"]["roc_auc"],
        ),
    )
    print("\n" + "=" * 70)
    print(f"CHAMPION MODEL SELECTED: {champion_name}")
    print("=" * 70)

    # Generate visualizations
    generate_evaluation_visualizations(evaluations, X_test, y_test, champion_name, ARTIFACTS_DIR)

    # Persist champion model pipeline
    champion_pipeline = evaluations[champion_name]["pipeline"]
    joblib.dump(champion_pipeline, MODEL_PATH)
    print(f"[+] Champion pipeline successfully saved to: {MODEL_PATH}")

    # Save comprehensive metadata JSON
    metadata = {
        "champion_model": champion_name,
        "best_hyperparameters": evaluations[champion_name]["best_params"],
        "cv_roc_auc": evaluations[champion_name]["cv_roc_auc"],
        "test_roc_auc": evaluations[champion_name]["metrics"]["roc_auc"],
        "test_pr_auc": evaluations[champion_name]["metrics"]["pr_auc"],
        "optimal_decision_threshold": evaluations[champion_name]["business"]["optimal_threshold"],
        "default_metrics": evaluations[champion_name]["metrics"],
        "optimal_threshold_metrics": evaluations[champion_name]["metrics_at_optimal_threshold"],
        "business_impact": {
            "customer_lifetime_value": CUSTOMER_LIFETIME_VALUE,
            "retention_offer_cost": RETENTION_OFFER_COST,
            "retention_success_rate": RETENTION_SUCCESS_RATE,
            "optimal_net_savings": evaluations[champion_name]["business"]["optimal_net_savings"],
            "default_net_savings": evaluations[champion_name]["business"]["default_net_savings"],
            "savings_lift_over_default": evaluations[champion_name]["business"]["savings_lift_over_default"],
            "optimal_targeted_customers": evaluations[champion_name]["business"]["optimal_targeted_customers"],
            "optimal_retained_customers": evaluations[champion_name]["business"]["optimal_retained_customers"],
            "optimal_roi_percent": evaluations[champion_name]["business"]["optimal_roi_percent"],
        },
        "all_models_summary": {
            name: {
                "cv_roc_auc": ev["cv_roc_auc"],
                "test_roc_auc": ev["metrics"]["roc_auc"],
                "test_pr_auc": ev["metrics"]["pr_auc"],
                "default_f1": ev["metrics"]["f1"],
                "optimal_threshold": ev["business"]["optimal_threshold"],
                "optimal_f1": ev["metrics_at_optimal_threshold"]["f1"],
                "optimal_net_savings": ev["business"]["optimal_net_savings"],
            }
            for name, ev in evaluations.items()
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Metadata written to: {METADATA_PATH}")

    return champion_name, metadata
