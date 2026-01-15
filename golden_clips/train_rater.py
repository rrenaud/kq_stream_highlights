#!/usr/bin/env python3
"""
Train a model to predict highlight ratings.

Usage:
    python -m golden_clips.train_rater              # Train model
    python -m golden_clips.train_rater --analyze-features  # Lasso feature selection

With only ~65 training samples, uses simple models with cross-validation.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from . import clip_features


MODEL_FILE = Path(__file__).parent / "rating_model.pkl"
FEATURE_IMPORTANCE_FILE = Path(__file__).parent / "feature_importance.json"
SELECTED_FEATURES_FILE = Path(__file__).parent / "selected_features.json"


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


def get_symmetric_index(idx: int, feature_names: list[str]) -> int | None:
    """Get the symmetric (opposite team) index for a feature.

    Returns None if the feature is team-agnostic.
    """
    name = feature_names[idx]

    # Base team-specific features (indices 19-26)
    # Pairs: blue_eggs_start/gold_eggs_start, etc.
    team_pairs = {
        19: 20, 20: 19,  # eggs_start
        21: 22, 22: 21,  # eggs_end
        23: 24, 24: 23,  # warriors_start
        25: 26, 26: 25,  # warriors_end
    }
    if idx in team_pairs:
        return team_pairs[idx]

    # State delta features (indices 27-78)
    # state_delta_0 to state_delta_19 = blue team (indices 27-46)
    # state_delta_20 to state_delta_39 = gold team (indices 47-66)
    # state_delta_40+ = neutral (maidens, map, snail, berries)
    if name.startswith("state_delta_"):
        delta_idx = int(name.split("_")[-1])
        if delta_idx < 20:
            # Blue -> Gold
            return idx + 20
        elif delta_idx < 40:
            # Gold -> Blue
            return idx - 20
        # Neutral features (maidens, map, snail, berries)
        return None

    # Team-agnostic features
    return None


def symmetrize_indices(indices: list[int], feature_names: list[str]) -> list[int]:
    """Add symmetric (opposite team) features for any team-specific features."""
    result = set(indices)
    for idx in indices:
        symmetric = get_symmetric_index(idx, feature_names)
        if symmetric is not None:
            result.add(symmetric)
    return sorted(result)


def analyze_features():
    """Run LassoCV to identify important features."""
    print("Loading training data...")
    X, y, clip_ids = clip_features.get_training_data()

    if len(y) < 10:
        print(f"ERROR: Need at least 10 rated clips, only have {len(y)}")
        return

    print(f"Training samples: {len(y)}")
    print(f"Feature dimension: {X.shape[1]}")
    print()

    feature_names = clip_features.ClipFeatures.feature_names()

    # Scale features for Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run LassoCV to find optimal alpha
    print("Running LassoCV to find optimal regularization...")
    alphas = np.logspace(-3, 1, 50)
    lasso_cv = LassoCV(cv=5, alphas=alphas, max_iter=10000)
    lasso_cv.fit(X_scaled, y)

    print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
    print()

    # Explore different alpha values to show feature count vs performance trade-off
    print("Feature count vs performance at different alpha values:")
    print("-" * 70)
    print(f"{'Alpha':>8s}  {'Features':>8s}  {'MAE':>7s}  {'Corr':>7s}  Top features")
    print("-" * 70)

    loo = LeaveOneOut()
    test_alphas = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    best_alpha = None
    best_mae = float('inf')

    for alpha in test_alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        nonzero = np.sum(lasso.coef_ != 0)

        if nonzero > 0:
            indices = np.where(lasso.coef_ != 0)[0]
            X_sub = X[:, indices]
            gb = GradientBoostingRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                min_samples_leaf=2, random_state=42
            )
            preds = cross_val_predict(gb, X_sub, y, cv=loo)
            mae = np.mean(np.abs(preds - y))
            corr = np.corrcoef(preds, y)[0, 1]

            # Get top 3 feature names
            coefs = [(feature_names[i], abs(lasso.coef_[i])) for i in indices]
            coefs.sort(key=lambda x: -x[1])
            top_names = ", ".join(n[:15] for n, _ in coefs[:3])

            print(f"{alpha:8.3f}  {nonzero:8d}  {mae:7.3f}  {corr:7.3f}  {top_names}")

            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
        else:
            print(f"{alpha:8.3f}  {nonzero:8d}  {'N/A':>7s}  {'N/A':>7s}")

    print("-" * 70)
    print()

    # Use best alpha from exploration
    chosen_alpha = best_alpha or 0.1
    print(f"Using alpha={chosen_alpha:.3f} (best MAE)")

    lasso_final = Lasso(alpha=chosen_alpha, max_iter=10000)
    lasso_final.fit(X_scaled, y)

    # Get non-zero coefficients
    nonzero_mask = lasso_final.coef_ != 0
    selected_indices = np.where(nonzero_mask)[0].tolist()
    selected_features = [(feature_names[i], float(lasso_final.coef_[i]))
                         for i in selected_indices]
    selected_features.sort(key=lambda x: -abs(x[1]))

    print(f"\nLasso selected {len(selected_features)} features:")
    print("-" * 60)
    for name, coef in selected_features:
        print(f"  {name:35s} coef: {coef:+.4f}")
    print("-" * 60)

    # Symmetrize features (add opposite team features)
    symmetric_indices = symmetrize_indices(selected_indices, feature_names)
    added_indices = [i for i in symmetric_indices if i not in selected_indices]

    print(f"\nSymmetrized to {len(symmetric_indices)} features (+{len(added_indices)} added):")
    print("-" * 60)
    for idx in symmetric_indices:
        marker = " " if idx in selected_indices else "+"
        print(f" {marker} {feature_names[idx]:35s} (index {idx})")
    print("-" * 60)
    print()

    # Compare full model vs reduced model vs symmetric model
    print("Final comparison:")
    print("-" * 60)

    # Full model (GradientBoosting)
    full_model = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        min_samples_leaf=2, random_state=42
    )
    full_preds = cross_val_predict(full_model, X, y, cv=loo)
    full_mae = np.mean(np.abs(full_preds - y))
    full_corr = np.corrcoef(full_preds, y)[0, 1]

    print(f"Full model (79 features):       MAE: {full_mae:.3f}  Corr: {full_corr:.3f}")

    # Reduced model using only Lasso-selected features
    if selected_indices:
        X_reduced = X[:, selected_indices]
        reduced_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            min_samples_leaf=2, random_state=42
        )
        reduced_preds = cross_val_predict(reduced_model, X_reduced, y, cv=loo)
        reduced_mae = np.mean(np.abs(reduced_preds - y))
        reduced_corr = np.corrcoef(reduced_preds, y)[0, 1]

        print(f"Lasso model ({len(selected_indices):2d} features):      MAE: {reduced_mae:.3f}  Corr: {reduced_corr:.3f}")

    # Symmetric model using symmetrized features
    if symmetric_indices:
        X_symmetric = X[:, symmetric_indices]
        symmetric_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            min_samples_leaf=2, random_state=42
        )
        symmetric_preds = cross_val_predict(symmetric_model, X_symmetric, y, cv=loo)
        symmetric_mae = np.mean(np.abs(symmetric_preds - y))
        symmetric_corr = np.corrcoef(symmetric_preds, y)[0, 1]

        print(f"Symmetric model ({len(symmetric_indices):2d} features):   MAE: {symmetric_mae:.3f}  Corr: {symmetric_corr:.3f}")
    print("-" * 60)

    # Save selected features (use symmetric version)
    save_data = {
        "chosen_alpha": float(chosen_alpha),
        "optimal_alpha_cv": float(lasso_cv.alpha_),
        "lasso_indices": selected_indices,
        "symmetric_indices": symmetric_indices,
        "selected_features": [{"name": feature_names[i], "index": i,
                               "from_lasso": i in selected_indices}
                              for i in symmetric_indices],
        "num_lasso_features": len(selected_indices),
        "num_symmetric_features": len(symmetric_indices),
        "total_features": len(feature_names),
    }

    print(f"\nSaving selected features to {SELECTED_FEATURES_FILE}...")
    with open(SELECTED_FEATURES_FILE, "w") as f:
        json.dump(save_data, f, indent=2)

    print("\nDone!")


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
    parser = argparse.ArgumentParser(description="Train highlight rating model")
    parser.add_argument("--analyze-features", action="store_true",
                        help="Run LassoCV feature selection analysis")
    args = parser.parse_args()

    if args.analyze_features:
        analyze_features()
    else:
        train_and_evaluate()
