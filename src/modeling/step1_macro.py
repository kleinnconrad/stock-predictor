import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from .base_pipeline import build_pipeline
from .diagnostics import calculate_ks_and_cutoff, calculate_cv_accuracy, generate_confusion_matrix, generate_lift_chart, get_confusion_matrix_dict
from sklearn.model_selection import TimeSeriesSplit
import os
import yaml

logger = logging.getLogger(__name__)

def execute_step1(df: pd.DataFrame, ticker: str = "UNKNOWN", n_features_out: int = 12) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes Step 1 modeling logic.
    
    Args:
        df (pd.DataFrame): The engineered DataFrame (historical rows with Target).
        n_features_out (int): Number of features to select.
        
    Returns:
        tuple: (metrics_dictionary, filtered_prediction_dataframe)
    """
    logger.info("Executing Step 1 Macro Model.")
    
    # Separate historical data where Target is known, vs most recent where Target is NaN
    train_df = df.dropna(subset=['Target'])
    pred_df = df[df['Target'].isna()]
    
    if train_df.empty:
        logger.warning("No training data available for Step 1.")
        return {}, df.iloc[0:0] # Return empty DF
        
    X_train = train_df.drop(columns=['Target', 'Future_Return'], errors='ignore')
    # Force strictly 0 or 1 to completely prevent any sklearn multiclass detection edge cases
    y_train = np.where(train_df['Target'] >= 0.5, 1, 0)
    
    X_pred = pred_df.drop(columns=['Target', 'Future_Return'], errors='ignore')
    
    # We must ensure we don't ask for more features than we have
    max_features = min(n_features_out, X_train.shape[1])
    pipeline = build_pipeline(n_features_out=max_features)
    
    # Generate Cross-Validation probabilities using strict TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    y_prob_cv_all = np.full(len(y_train), np.nan)
    
    try:
        # Manual loop to prevent data leakage in time series
        for train_index, test_index in tscv.split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold = y_train[train_index]
            
            pipeline.fit(X_train_fold, y_train_fold)
            # Store predictions at the exact index of the test set
            y_prob_cv_all[test_index] = pipeline.predict_proba(X_test_fold)[:, 1]
            
    except Exception as e:
        logger.error(f"Failed to generate TimeSeries CV probabilities: {e}")
        pipeline.fit(X_train, y_train)
        y_prob_cv_all = pipeline.predict_proba(X_train)[:, 1]
    
    # Remove NaNs (affects the first training fold which has no test data)
    valid_indices = ~np.isnan(y_prob_cv_all)
    y_prob_cv_clean = y_prob_cv_all[valid_indices]
    y_train_clean = y_train[valid_indices]
    
    # Calculate KS and Cutoff on clean data
    ks_stat, ks_cutoff = calculate_ks_and_cutoff(y_train_clean, y_prob_cv_clean)
    cv_accuracy = calculate_cv_accuracy(y_train_clean, y_prob_cv_clean, ks_cutoff)
    cv_confusion_matrix = get_confusion_matrix_dict(y_train_clean, y_prob_cv_clean, ks_cutoff)
    
    diag_dir = os.path.join('outputs', 'diagnostics', ticker)
    os.makedirs(diag_dir, exist_ok=True)
    
    # Generate Visual Artifacts for Cross Validation
    generate_confusion_matrix(y_train_clean, y_prob_cv_clean, ks_cutoff, os.path.join(diag_dir, f"{ticker}_cv_confusion_matrix.png"))
    generate_lift_chart(y_train_clean, y_prob_cv_clean, 10, os.path.join(diag_dir, f"{ticker}_cv_lift_chart.png"))
    
    # Now fit on the entire historical dataset to get the final model weights for prediction
    pipeline.fit(X_train, y_train)
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
    
    # Load minimum CV accuracy from settings
    try:
        with open('config/settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
            min_cv_accuracy = float(settings.get('min_cv_accuracy', 0.65))
    except Exception as e:
        logger.warning(f"Failed to load min_cv_accuracy from settings.yaml: {e}. Defaulting to 0.65")
        min_cv_accuracy = 0.65

    # Determine predicted class for the most recent date
    latest_pred_class = "NOT_UP"
    if len(y_pred_prob) > 0:
        latest_prob = y_pred_prob[-1]
        if cv_accuracy < min_cv_accuracy:
            logger.info(f"Failed Step 1 for {ticker}: CV Accuracy ({cv_accuracy:.2f}) is below the {min_cv_accuracy} threshold.")
        elif latest_prob >= ks_cutoff:
            latest_pred_class = "UP"
            
    # Also filter the df dates where the *historical* model prediction >= ks_cutoff to pass to Step 2
    # Wait, the spec says: "Cascade filter Step 1 dates where y_prob >= ks_optimal_cutoff to Step 2."
    # We apply the model to the ENTIRE dataset (historical + pred) to get filter mask.
    all_X = df.drop(columns=['Target', 'Future_Return'], errors='ignore')
    all_prob = pipeline.predict_proba(all_X)[:, 1]
    filtered_df = df[all_prob >= ks_cutoff]
            
    metrics = {
        "cv_accuracy": float(cv_accuracy),
        "ks_cutoff": float(ks_cutoff),
        "latest_prob": float(latest_prob) if len(y_pred_prob) > 0 else None,
        "predicted_class": latest_pred_class,
        "selected_predictors_and_weights": selected_predictors_and_weights,
        "feature_diagnostics": feature_diagnostics,
        "cv_confusion_matrix": cv_confusion_matrix
    }
    
    mat_dir = os.path.join('outputs', 'matrices')
    os.makedirs(mat_dir, exist_ok=True)
    train_df.to_csv(os.path.join(mat_dir, f"{ticker}_step1_train.csv"))
    pred_df.to_csv(os.path.join(mat_dir, f"{ticker}_step1_pred.csv"))
    
    return metrics, filtered_df
