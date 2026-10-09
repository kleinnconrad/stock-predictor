import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from .base_pipeline import build_pipeline, purged_time_series_cv, InsufficientHistoryError
from sklearn.metrics import accuracy_score
from .diagnostics import (confusion_counts, generate_confusion_matrix, generate_lift_chart, ks_cutoff_or_none,
                          plot_confusion_matrix, walk_forward_predictions)
import os
from config.settings import load_settings

logger = logging.getLogger(__name__)

CV_STATUS_OK = "ok"


def _rejected_metrics(cv_status: str, latest_prob: Optional[float] = None) -> Dict[str, Any]:
    """
    Builds the Step 1 metrics for a ticker that cannot be validated.

    Args:
        cv_status (str): Why the ticker could not be validated.
        latest_prob (Optional[float]): Latest predicted probability, if one exists.

    Returns:
        Dict[str, Any]: Metrics with predicted_class NOT_UP and no invented scores.
    """
    return {
        "cv_status": cv_status,
        "cv_accuracy": None,
        "ks_cutoff": None,
        "latest_prob": latest_prob,
        "predicted_class": "NOT_UP",
        "selected_predictors_and_weights": {},
        "feature_diagnostics": {},
        "cv_confusion_matrix": {},
        "cm_rule_passed": False,
    }


def execute_step1(df: pd.DataFrame, ticker: str = "UNKNOWN") -> Dict[str, Any]:
    """
    Executes Step 1 modeling logic.

    The model is validated with an expanding-window TimeSeriesSplit whose folds are
    separated by `horizon_days` rows (purged), because each target looks `horizon_days`
    into the future. Each test fold after the first is classified with the KS cutoff
    learned on the earlier folds (walk-forward), and only those folds are scored. The
    cutoff applied to the live prediction maximizes KS over all out-of-fold predictions.
    If the ticker cannot be validated (too little history, a training fold with a single
    class, a failing fit, or a cutoff without discriminatory power), it is rejected with
    predicted_class NOT_UP and a `cv_status` explaining why; no score is invented.

    Model parameters (horizon_days, features_to_select, anova_k, cv_splits, sfs_cv_splits,
    min_cv_train_rows, quantiles, min_cv_accuracy) are read from config/settings.yaml.

    Args:
        df (pd.DataFrame): The engineered DataFrame (historical rows with Target).
        ticker (str): The ticker being modeled; used to name the diagnostic artifacts.

    Returns:
        Dict[str, Any]: The Step 1 metrics, including `predicted_class` and `cv_status`.
    """
    logger.info("Executing Step 1 Macro Model.")
    settings = load_settings()
    horizon_days = int(settings['horizon_days'])
    n_features_out = int(settings['features_to_select'])
    anova_k = int(settings['anova_k'])
    cv_splits = int(settings['cv_splits'])
    sfs_cv_splits = int(settings['sfs_cv_splits'])
    min_cv_train_rows = int(settings['min_cv_train_rows'])
    quantiles = int(settings['quantiles'])
    min_cv_accuracy = float(settings['min_cv_accuracy'])

    # Separate historical data where Target is known, vs most recent where Target is NaN
    train_df = df.dropna(subset=['Target'])
    pred_df = df[df['Target'].isna()]

    if train_df.empty:
        logger.warning(f"No training data available for Step 1 ({ticker}).")
        return _rejected_metrics("insufficient_history")

    X_train = train_df.drop(columns=['Target', 'Future_Return'], errors='ignore')
    # Force strictly 0 or 1 to completely prevent any sklearn multiclass detection edge cases
    y_train = np.where(train_df['Target'] >= 0.5, 1, 0)

    X_pred = pred_df.drop(columns=['Target', 'Future_Return'], errors='ignore')

    if len(np.unique(y_train)) < 2:
        logger.info(f"Failed Step 1 for {ticker}: the target has a single class in the training history.")
        return _rejected_metrics("single_class_target")

    # We must ensure we don't ask for more features than we have
    max_features = min(n_features_out, X_train.shape[1])

    def fit_pipeline(X: pd.DataFrame, y: np.ndarray) -> Tuple[Pipeline, bool]:
        # The SFS selection CV is purged as well where the fold sizes allow it
        sfs_cv, sfs_purged = purged_time_series_cv(
            len(X), sfs_cv_splits, gap=horizon_days, min_train_rows=min_cv_train_rows, allow_unpurged=True)
        pipeline = build_pipeline(n_features_out=max_features, anova_k=anova_k, sfs_cv=sfs_cv)
        pipeline.fit(X, y)
        return pipeline, sfs_purged

    # Generate out-of-fold probabilities with a purged, expanding TimeSeriesSplit
    try:
        outer_cv, _ = purged_time_series_cv(
            len(X_train), cv_splits, gap=horizon_days, min_train_rows=min_cv_train_rows)
    except InsufficientHistoryError as e:
        logger.info(f"Failed Step 1 for {ticker}: {e}")
        return _rejected_metrics("insufficient_history")

    fold_results = []
    try:
        for train_index, test_index in outer_cv.split(X_train):
            y_train_fold = y_train[train_index]
            if len(np.unique(y_train_fold)) < 2:
                raise ValueError("a training fold contains a single class")
            fold_pipeline, _ = fit_pipeline(X_train.iloc[train_index], y_train_fold)
            fold_probs = fold_pipeline.predict_proba(X_train.iloc[test_index])[:, 1]
            fold_results.append((y_train[test_index], fold_probs))
    except Exception as e:
        logger.info(f"Failed Step 1 for {ticker}: cross-validation failed ({e}).")
        return _rejected_metrics(f"cv_failed: {e}")

    # Score folds 2..k with cutoffs learned walk-forward; derive the production cutoff from all folds
    y_oof = np.concatenate([y for y, _ in fold_results])
    p_oof = np.concatenate([p for _, p in fold_results])
    walk_forward = walk_forward_predictions(fold_results)
    production_cutoff = ks_cutoff_or_none(y_oof, p_oof)
    if walk_forward is None or production_cutoff is None:
        logger.info(f"Failed Step 1 for {ticker}: the KS cutoff has no discriminatory power.")
        return _rejected_metrics("degenerate_cutoff")

    y_scored, y_pred_scored = walk_forward
    ks_stat, ks_cutoff = production_cutoff
    cv_accuracy = accuracy_score(y_scored, y_pred_scored)
    cv_confusion_matrix = confusion_counts(y_scored, y_pred_scored)

    diag_dir = os.path.join('outputs', 'diagnostics', ticker)
    os.makedirs(diag_dir, exist_ok=True)

    # Generate Visual Artifacts for Cross Validation
    plot_confusion_matrix(y_scored, y_pred_scored, 'Walk-forward CV Confusion Matrix',
                          os.path.join(diag_dir, f"{ticker}_cv_confusion_matrix.png"))
    generate_lift_chart(y_oof, p_oof, quantiles, os.path.join(diag_dir, f"{ticker}_cv_lift_chart.png"))

    # Now fit on the entire historical dataset to get the final model weights for prediction
    pipeline, sfs_cv_purged = fit_pipeline(X_train, y_train)
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]

    # Generate Visual Artifacts for Full Training Set
    generate_confusion_matrix(y_train, y_prob_train, ks_cutoff, os.path.join(diag_dir, f"{ticker}_train_confusion_matrix.png"))

    # Predict probabilities for the most recent data (the real prediction)
    if not X_pred.empty:
        y_pred_prob = pipeline.predict_proba(X_pred)[:, 1]
    else:
        y_pred_prob = np.array([])

    # Feature Extraction Logic
    var_thresh = pipeline.named_steps['var_thresh']
    anova = pipeline.named_steps['anova']
    sfs = pipeline.named_steps['sfs']
    clf = pipeline.named_steps['clf']

    # 0. Get Variance Threshold mask
    var_mask = var_thresh.get_support()
    var_features = X_train.columns[var_mask]

    # 1. Get ANOVA mask and filter
    anova_mask = anova.get_support()
    anova_features = var_features[anova_mask]

    # 2. Get SFS mask and filter ANOVA features
    sfs_mask = sfs.get_support()
    final_features = anova_features[sfs_mask].tolist()

    # 3. Extract Logistic Regression weights
    weights = clf.coef_[0].tolist()

    selected_predictors_and_weights = {
        feature: float(weight) for feature, weight in zip(final_features, weights)
    }

    # 4. Generate Feature Diagnostics
    fetched_features = X_train.columns.tolist()
    removed_by_anova = list(set(fetched_features) - set(anova_features.tolist()))
    removed_by_sfs = list(set(anova_features.tolist()) - set(final_features))

    feature_diagnostics = {
        "fetched_features": fetched_features,
        "removed_by_anova": removed_by_anova,
        "removed_by_sfs": removed_by_sfs
    }

    # Evaluate Confusion Matrix Rule: TP > FN and TN > FP
    cm_rule_passed = (cv_confusion_matrix["TP"] > cv_confusion_matrix["FN"]) and \
                     (cv_confusion_matrix["TN"] > cv_confusion_matrix["FP"])

    # Determine predicted class for the most recent date
    latest_pred_class = "NOT_UP"
    latest_prob = float(y_pred_prob[-1]) if len(y_pred_prob) > 0 else None
    if latest_prob is not None:
        if cv_accuracy < min_cv_accuracy:
            logger.info(f"Failed Step 1 for {ticker}: CV Accuracy ({cv_accuracy:.2f}) is below the {min_cv_accuracy} threshold.")
        elif not cm_rule_passed:
            logger.info(f"Failed Step 1 for {ticker}: Confusion Matrix rule failed ({cv_confusion_matrix}).")
        elif latest_prob >= ks_cutoff:
            latest_pred_class = "UP"

    metrics = {
        "cv_status": CV_STATUS_OK,
        "cv_accuracy": float(cv_accuracy),
        "cv_scored_rows": int(len(y_scored)),
        "ks_stat": float(ks_stat),
        "ks_cutoff": float(ks_cutoff),
        "latest_prob": latest_prob,
        "predicted_class": latest_pred_class,
        "selected_predictors_and_weights": selected_predictors_and_weights,
        "feature_diagnostics": feature_diagnostics,
        "cv_confusion_matrix": cv_confusion_matrix,
        "cm_rule_passed": cm_rule_passed,
        "sfs_cv_purged": sfs_cv_purged,
    }

    mat_dir = os.path.join('outputs', 'matrices')
    os.makedirs(mat_dir, exist_ok=True)
    train_df.to_csv(os.path.join(mat_dir, f"{ticker}_step1_train.csv"))
    pred_df.to_csv(os.path.join(mat_dir, f"{ticker}_step1_pred.csv"))

    return metrics
