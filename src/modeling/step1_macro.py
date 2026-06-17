import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from .base_pipeline import build_pipeline
from .diagnostics import calculate_ks_and_cutoff, calculate_cv_accuracy
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit

logger = logging.getLogger(__name__)

def execute_step1(df: pd.DataFrame, n_features_out: int = 12) -> Tuple[Dict[str, Any], pd.DataFrame]:
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
    y_train = train_df['Target'].values
    
    X_pred = pred_df.drop(columns=['Target', 'Future_Return'], errors='ignore')
    
    # We must ensure we don't ask for more features than we have
    max_features = min(n_features_out, X_train.shape[1])
    pipeline = build_pipeline(n_features_out=max_features)
    
    # Generate Cross-Validation probabilities using TimeSeriesSplit to avoid lookahead bias
    cv = TimeSeriesSplit(n_splits=3)
    
    # We need proba, cross_val_predict can return it if method='predict_proba'
    try:
        y_prob_cv = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    except Exception as e:
        logger.error(f"Failed to generate CV probabilities: {e}")
        # Fallback if there's an issue with CV splitting due to small dataset size
        pipeline.fit(X_train, y_train)
        y_prob_cv = pipeline.predict_proba(X_train)[:, 1]

    # Calculate KS and Cutoff
    ks_stat, ks_cutoff = calculate_ks_and_cutoff(y_train, y_prob_cv)
    cv_accuracy = calculate_cv_accuracy(y_train, y_prob_cv, ks_cutoff)
    
    # Now fit on the entire historical dataset to get the final model weights for prediction
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities for the most recent data (the real prediction)
    if not X_pred.empty:
        y_pred_prob = pipeline.predict_proba(X_pred)[:, 1]
    else:
        y_pred_prob = np.array([])
        
    # Feature Extraction Logic
    anova = pipeline.named_steps['anova']
    sfs = pipeline.named_steps['sfs']
    clf = pipeline.named_steps['clf']
    
    # 1. Get ANOVA mask and filter original features
    anova_mask = anova.get_support()
    anova_features = X_train.columns[anova_mask]
    
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
    
    # Determine predicted class for the most recent date
    latest_pred_class = "NOT_UP"
    if len(y_pred_prob) > 0:
        latest_prob = y_pred_prob[-1]
        if latest_prob >= ks_cutoff:
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
        "predicted_class": latest_pred_class,
        "selected_predictors_and_weights": selected_predictors_and_weights,
        "feature_diagnostics": feature_diagnostics
    }
    
    return metrics, filtered_df
