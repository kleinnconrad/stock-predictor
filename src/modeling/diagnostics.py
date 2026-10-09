import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

def calculate_ks_and_cutoff(y_true: np.ndarray, y_prob_1: np.ndarray):
    """
    Finds the threshold that maximizes the KS statistic: max(TPR - FPR).
    
    Args:
        y_true (np.ndarray): True binary labels.
        y_prob_1 (np.ndarray): Predicted probabilities for the positive class.
        
    Returns:
        tuple: (ks_stat, optimal_cutoff)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_1)
    ks_stats = tpr - fpr
    max_idx = np.argmax(ks_stats)
    ks_stat = ks_stats[max_idx]
    optimal_cutoff = thresholds[max_idx]
    return ks_stat, optimal_cutoff

def ks_cutoff_or_none(y_true: np.ndarray, y_prob_1: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Returns the KS statistic and cutoff, or None if they cannot discriminate.

    The cutoff is unusable when the labels contain a single class, when the maximum
    of TPR - FPR is not positive, or when the cutoff is not finite (roc_curve returns
    +inf as its first threshold, which would classify every row as negative).

    Args:
        y_true (np.ndarray): True binary labels.
        y_prob_1 (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        Optional[Tuple[float, float]]: (ks_stat, cutoff) or None.
    """
    if len(np.unique(y_true)) < 2:
        return None
    ks_stat, cutoff = calculate_ks_and_cutoff(y_true, y_prob_1)
    if not np.isfinite(cutoff) or ks_stat <= 0:
        return None
    return float(ks_stat), float(cutoff)

def walk_forward_predictions(fold_results: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Classifies each test fold with a KS cutoff learned only on the folds before it.

    Choosing the cutoff on the same predictions that are then scored inflates the
    reported accuracy. Here fold j is classified with the cutoff that maximizes KS on
    the out-of-fold predictions of folds 1..j-1, so the first fold is only used to
    learn a cutoff and is not scored.

    Args:
        fold_results (List[Tuple[np.ndarray, np.ndarray]]): (y_true, y_prob) per test
            fold in chronological order.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (y_true, y_pred) of the scored folds,
        or None if any required cutoff is degenerate or fewer than two folds exist.
    """
    if len(fold_results) < 2:
        return None
    y_scored, y_pred_scored = [], []
    for j in range(1, len(fold_results)):
        hist_y = np.concatenate([y for y, _ in fold_results[:j]])
        hist_p = np.concatenate([p for _, p in fold_results[:j]])
        ks = ks_cutoff_or_none(hist_y, hist_p)
        if ks is None:
            return None
        y_j, p_j = fold_results[j]
        y_scored.append(y_j)
        y_pred_scored.append((p_j >= ks[1]).astype(int))
    return np.concatenate(y_scored), np.concatenate(y_pred_scored)

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns a dictionary of TP, FP, TN, FN for binary labels and predictions.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: str):
    """
    Generates and saves a confusion matrix plot from binary predictions.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def generate_confusion_matrix(y_true: np.ndarray, y_prob_1: np.ndarray, cutoff: float, output_path: str):
    """
    Generates and saves a confusion matrix plot.
    """
    y_pred = (y_prob_1 >= cutoff).astype(int)
    plot_confusion_matrix(y_true, y_pred, f'Confusion Matrix (Cutoff: {cutoff:.3f})', output_path)

def generate_lift_chart(y_true: np.ndarray, y_prob_1: np.ndarray, quantiles: int, output_path: str):
    """
    Generates and saves a 10-quantile lift chart.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob_1})
    
    # Safe binning using duplicates='drop'
    df['quantile'] = pd.qcut(df['y_prob'], q=quantiles, labels=False, duplicates='drop')
    
    lift = df.groupby('quantile')['y_true'].mean() / df['y_true'].mean()
    
    plt.figure(figsize=(8, 5))
    lift.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Lift Chart by Quantile')
    plt.xlabel('Probability Quantile')
    plt.ylabel('Lift')
    plt.axhline(1.0, color='red', linestyle='--')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
