"""
Model classifiers — LightGBM, XGBoost, Logistic Regression.

No internal train/test splitting. Takes pre-split data ONLY.
Returns trained model with predict_proba() capability.
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    config: Dict[str, Any],
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Any:
    """
    Train a classifier on pre-split data.

    Args:
        X_train: Feature matrix (n_samples, n_features).
        y_train: Label vector (n_samples,).
        model_type: 'lightgbm', 'xgboost', or 'logistic_regression'.
        config: Model hyperparameters from config.yaml.
        seed: Random seed.
        feature_names: Feature names for LightGBM.

    Returns:
        Trained model with .predict_proba(X) method.
    """
    n_classes = len(np.unique(y_train[~np.isnan(y_train)]))

    if model_type == "lightgbm":
        params = {
            "learning_rate": config.get("learning_rate", 0.05),
            "n_estimators": config.get("n_estimators", 400),
            "max_depth": config.get("max_depth", 3),
            "num_leaves": config.get("num_leaves", 8),
            "min_child_samples": config.get("min_child_samples", 200),
            "subsample": config.get("subsample", 0.7),
            "colsample_bytree": config.get("colsample_bytree", 0.7),
            "reg_alpha": config.get("reg_alpha", 1.0),
            "reg_lambda": config.get("reg_lambda", 5.0),
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": -1,
        }
        if n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
        else:
            params["objective"] = "binary"

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            feature_name=feature_names if feature_names else "auto",
        )
        return model

    elif model_type == "xgboost":
        params = {
            "learning_rate": config.get("learning_rate", 0.05),
            "n_estimators": config.get("n_estimators", 400),
            "max_depth": config.get("max_depth", 6),
            "min_child_weight": config.get("min_child_weight", 5),
            "subsample": config.get("subsample", 0.7),
            "colsample_bytree": config.get("colsample_bytree", 0.7),
            "reg_alpha": config.get("reg_alpha", 1.0),
            "reg_lambda": config.get("reg_lambda", 5.0),
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": 0,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        if n_classes > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
        else:
            params["objective"] = "binary:logistic"

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model

    elif model_type == "logistic_regression":
        params = {
            "penalty": config.get("penalty", "l2"),
            "C": config.get("C", 1.0),
            "solver": config.get("solver", "lbfgs"),
            "max_iter": config.get("max_iter", 1000),
            "random_state": seed,
            "n_jobs": 1,
        }

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])
        model.fit(X_train, y_train)
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Get class probabilities. Shape: (n_samples, n_classes).
    """
    return model.predict_proba(X)
