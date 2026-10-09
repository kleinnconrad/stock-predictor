import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # charts are only saved to files; interactive backends crash outside the main thread
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Optional, Tuple

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

def classify_with_own_cutoff(y_fit: np.ndarray, p_fit: np.ndarray,
                             p_apply: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """
    Classifies a model's predictions with the KS cutoff learned on the same model's training rows.

    Refitted logistic regressions put their probabilities on very different scales, so a
    cutoff is only meaningful for the model whose probabilities it was computed on. Learning
    it on the training rows also keeps the test rows out of the threshold choice.

    Args:
        y_fit (np.ndarray): Labels of the model's training rows.
        p_fit (np.ndarray): The model's probabilities on its training rows.
        p_apply (np.ndarray): The same model's probabilities on the rows to classify.

    Returns:
        Optional[Tuple[np.ndarray, float]]: (binary predictions, cutoff), or None if the
        training predictions yield no usable cutoff.
    """
    ks = ks_cutoff_or_none(y_fit, p_fit)
    if ks is None:
        return None
    return (np.asarray(p_apply) >= ks[1]).astype(int), ks[1]

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
