import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

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

def calculate_cv_accuracy(y_true: np.ndarray, y_prob_1: np.ndarray, cutoff: float) -> float:
    """
    Applies the KS cutoff to convert probabilities to binary classes, then calculates accuracy.
    
    Args:
        y_true (np.ndarray): True labels.
        y_prob_1 (np.ndarray): Predicted probabilities.
        cutoff (float): The optimal cutoff threshold.
        
    Returns:
        float: Accuracy score.
    """
    y_pred = (y_prob_1 >= cutoff).astype(int)
    return accuracy_score(y_true, y_pred)

def generate_confusion_matrix(y_true: np.ndarray, y_prob_1: np.ndarray, cutoff: float, output_path: str):
    """
    Generates and saves a confusion matrix plot.
    """
    y_pred = (y_prob_1 >= cutoff).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Cutoff: {cutoff:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

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
