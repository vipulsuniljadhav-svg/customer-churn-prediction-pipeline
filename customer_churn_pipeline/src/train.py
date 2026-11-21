"""Model definition, hyperparameter tuning with cross-validation, and candidate training."""

from typing import Any, Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import CV_FOLDS, RANDOM_STATE
from src.preprocessing import build_pipeline


def get_candidate_models() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """Return dictionary of candidate models and their hyperparameter grids for tuning."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "classifier__C": [0.05, 0.2, 1.0, 5.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [6, 10, None],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_depth": [3, 4],
            },
        ),
    }
    return models


def train_and_tune_model(
    model_name: str,
    base_estimator: Any,
    param_grid: Dict[str, list],
    X_train,
    y_train,
    cv_folds: int = CV_FOLDS,
    scoring: str = "roc_auc",
) -> Tuple[Any, Dict[str, Any]]:
    """Tune a model pipeline using Stratified K-Fold cross validation and GridSearchCV."""
    pipeline = build_pipeline(base_estimator)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    tuning_summary = {
        "model_name": model_name,
        "best_params": grid_search.best_params_,
        "best_cv_roc_auc": float(grid_search.best_score_),
    }

    return grid_search.best_estimator_, tuning_summary


def train_all_candidates(X_train, y_train) -> Dict[str, Dict[str, Any]]:
    """Train and tune all candidate models, returning their fitted pipelines and CV summaries."""
    candidates = get_candidate_models()
    trained_models = {}

    for name, (estimator, params) in candidates.items():
        print(f"[*] Training and tuning {name} with {CV_FOLDS}-fold cross validation...")
        best_pipeline, summary = train_and_tune_model(
            name, estimator, params, X_train, y_train
        )
        trained_models[name] = {
            "pipeline": best_pipeline,
            "summary": summary,
        }
        print(f"    -> Best CV ROC-AUC: {summary['best_cv_roc_auc']:.4f}")
        print(f"    -> Best Params: {summary['best_params']}")

    return trained_models
