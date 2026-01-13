#!/usr/bin/env python3
"""
Train a model to predict highlight ratings.

Usage:
    python golden_clips/train_rater.py

With only ~12 training samples, uses simple models with cross-validation.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from . import clip_features


MODEL_FILE = Path(__file__).parent / "rating_model.pkl"
FEATURE_IMPORTANCE_FILE = Path(__file__).parent / "feature_importance.json"


def train_and_evaluate():
    """Train models and evaluate with leave-one-out cross-validation."""
    print("Loading training data...")
    X, y, clip_ids = clip_features.get_training_data()

    if len(y) < 5:
        print(f"ERROR: Need at least 5 rated clips, only have {len(y)}")
        return None

    print(f"Training samples: {len(y)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Rating distribution: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.2f}")
    print()

    # Define models to try
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=2,
            random_state=42
        ),
    }

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    results = {}

    print("Evaluating models with leave-one-out cross-validation:")
    print("-" * 60)

    for name, model in models.items():
        predictions = cross_val_predict(model, X, y, cv=loo)

        # Calculate metrics
        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(np.mean((predictions - y) ** 2))
        corr = np.corrcoef(predictions, y)[0, 1] if len(y) > 1 else 0

        results[name] = {
            "mae": mae,
            "rmse": rmse,
            "correlation": corr,
            "predictions": predictions,
        }

        print(f"{name:20s} MAE: {mae:.3f}  RMSE: {rmse:.3f}  Corr: {corr:.3f}")

    print("-" * 60)

    # Select best model based on MAE
    best_name = min(results.keys(), key=lambda k: results[k]["mae"])
    print(f"\nBest model: {best_name}")

    # Train final model on all data
    print(f"\nTraining final {best_name} model on all data...")
    final_model = models[best_name]
    final_model.fit(X, y)

    # Show predictions vs actual
    print("\nLeave-one-out predictions:")
    predictions = results[best_name]["predictions"]
    for i, (clip_id, actual, pred) in enumerate(zip(clip_ids, y, predictions)):
        error = "OK" if abs(pred - actual) < 1 else "MISS"
        print(f"  {clip_id[:20]:20s} Actual: {actual:.0f}  Pred: {pred:.2f}  [{error}]")

    # Calculate and save feature importance
    feature_names = clip_features.ClipFeatures.feature_names()
    importance = get_feature_importance(final_model, feature_names)

    print(f"\nTop 15 most important features:")
    for name, imp in importance[:15]:
        print(f"  {name:30s} {imp:.4f}")

    # Save model
    print(f"\nSaving model to {MODEL_FILE}...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "model": final_model,
            "model_name": best_name,
            "feature_names": feature_names,
            "train_samples": len(y),
            "metrics": {k: {m: float(v) for m, v in results[k].items() if m != "predictions"}
                       for k in results},
        }, f)

    # Save feature importance
    with open(FEATURE_IMPORTANCE_FILE, "w") as f:
        json.dump([{"name": n, "importance": float(i)} for n, i in importance], f, indent=2)

    return final_model


def get_feature_importance(model, feature_names: list[str]) -> list[tuple[str, float]]:
    """Extract feature importance from model."""
    if hasattr(model, "named_steps"):
        # Pipeline - get from inner model
        inner = model.named_steps.get("model")
        if hasattr(inner, "coef_"):
            importance = np.abs(inner.coef_)
        elif hasattr(inner, "feature_importances_"):
            importance = inner.feature_importances_
        else:
            importance = np.zeros(len(feature_names))
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
    else:
        importance = np.zeros(len(feature_names))

    paired = list(zip(feature_names, importance))
    return sorted(paired, key=lambda x: -x[1])


def load_model():
    """Load trained model."""
    if not MODEL_FILE.exists():
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def predict_rating(features: clip_features.ClipFeatures) -> float:
    """Predict rating for a clip given its features."""
    model_data = load_model()
    if model_data is None:
        return 2.5  # Default middle rating

    model = model_data["model"]
    X = features.to_feature_vector().reshape(1, -1)
    prediction = model.predict(X)[0]

    # Clip to valid range [1, 4]
    return max(1.0, min(4.0, prediction))


if __name__ == "__main__":
    train_and_evaluate()
