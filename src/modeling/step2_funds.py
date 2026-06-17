import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from .base_pipeline import build_pipeline
from .diagnostics import calculate_ks_and_cutoff, calculate_cv_accuracy
from sklearn.model_selection import cross_val_predict, StratifiedKFold

logger = logging.getLogger(__name__)

def execute_step2(df: pd.DataFrame, n_features_out: int = 12) -> Tuple[Dict[str, Any], str]:
    """
    Executes Step 2 fundamental modeling logic.
    
    Args:
        df (pd.DataFrame): The cascaded DataFrame from Step 1 (containing fundamentals).
        n_features_out (int): Number of features to select.
        
    Returns:
        tuple: (metrics_dictionary, final_prediction_class)
    """
    logger.info("Executing Step 2 Fundamental Model.")
    
    # Same target logic as Step 1
    train_df = df.dropna(subset=['Target'])
    pred_df = df[df['Target'].isna()]
    
    if train_df.empty:
        logger.warning("No training data available for Step 2.")
        return {}, "NOT_UP"
        
    X_train = train_df.drop(columns=['Target', 'Future_Return'], errors='ignore')
    # Force strictly 0 or 1
    y_train = np.where(train_df['Target'] >= 0.5, 1, 0)
    
    # Check if we have any variance at all. Due to backfilling YF data, older periods
    # might be completely constant, causing VarianceThreshold to drop all features.
    if X_train.empty or X_train.var().max() == 0:
        logger.error("No variance found in fundamental features (all constant). Skipping Step 2.")
        return {}, "NOT_UP"
    
    X_pred = pred_df.drop(columns=['Target', 'Future_Return'], errors='ignore')
    
    # We must ensure we don't ask for more features than we have
    max_features = min(n_features_out, X_train.shape[1])
    if max_features == 0:
        logger.warning("No features remaining for Step 2.")
        return {}, "NOT_UP"
        
    pipeline = build_pipeline(n_features_out=max_features)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    try:
        y_prob_cv = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    except Exception as e:
        logger.warning(f"Failed to generate CV probabilities for Step 2, falling back to full fit. Reason: {e}")
        try:
            pipeline.fit(X_train, y_train)
            y_prob_cv = pipeline.predict_proba(X_train)[:, 1]
        except Exception as e2:
            logger.error(f"Failed to fit fallback pipeline for Step 2. Reason: {e2}")
            return {}, "NOT_UP"

    # Calculate KS and Cutoff
    ks_stat, ks_cutoff = calculate_ks_and_cutoff(y_train, y_prob_cv)
    cv_accuracy = calculate_cv_accuracy(y_train, y_prob_cv, ks_cutoff)
    
    # Now fit on the entire historical dataset to get the final model weights for prediction
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Failed to fit final pipeline for Step 2. Reason: {e}")
        return {}, "NOT_UP"
    
    if not X_pred.empty:
        y_pred_prob = pipeline.predict_proba(X_pred)[:, 1]
    else:
        y_pred_prob = np.array([])
        
    # Feature Extraction
    var_thresh = pipeline.named_steps['var_thresh']
    anova = pipeline.named_steps['anova']
    sfs = pipeline.named_steps['sfs']
    clf = pipeline.named_steps['clf']
    
    var_mask = var_thresh.get_support()
    var_features = X_train.columns[var_mask]
    
    anova_mask = anova.get_support()
    anova_features = var_features[anova_mask]
    
    sfs_mask = sfs.get_support()
    final_features = anova_features[sfs_mask].tolist()
    
    weights = clf.coef_[0].tolist()
    
    selected_predictors_and_weights = {
        feature: float(weight) for feature, weight in zip(final_features, weights)
    }
    
    # Generate Feature Diagnostics
    fetched_features = X_train.columns.tolist()
    removed_by_anova = list(set(fetched_features) - set(anova_features.tolist()))
    removed_by_sfs = list(set(anova_features.tolist()) - set(final_features))
    
    feature_diagnostics = {
        "fetched_features": fetched_features,
        "removed_by_anova": removed_by_anova,
        "removed_by_sfs": removed_by_sfs
    }
    
    latest_pred_class = "NOT_UP"
    if len(y_pred_prob) > 0:
        latest_prob = y_pred_prob[-1]
        if latest_prob >= ks_cutoff:
            latest_pred_class = "UP"
            
    metrics = {
        "cv_accuracy": float(cv_accuracy),
        "ks_cutoff": float(ks_cutoff),
        "predicted_class": latest_pred_class,
        "selected_predictors_and_weights": selected_predictors_and_weights,
        "feature_diagnostics": feature_diagnostics
    }
    
    return metrics, latest_pred_class
