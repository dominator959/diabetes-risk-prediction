"""
evaluator.py
------------
Model evaluation utilities. Computes and formats standard
classification metrics and generates evaluation charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute a full set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    metrics = {
        'accuracy' : round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall'   : round(recall_score(y_true, y_pred), 4),
        'f1_score' : round(f1_score(y_true, y_pred), 4),
    }
    if y_prob is not None:
        metrics['roc_auc'] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a comparison table from multiple model result dicts.

    Parameters
    ----------
    results : dict
        Keys are model names, values are metric dicts from
        compute_metrics().

    Returns
    -------
    pd.DataFrame
        Formatted comparison table sorted by F1 score.
    """
    df = pd.DataFrame(results).T
    df = df.sort_values('f1_score', ascending=False)
    return df


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    save_path: str = None
):
    """
    Plot a styled confusion matrix for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Name of the model for the chart title.
    save_path : str, optional
        File path to save the chart. If None, chart is not saved.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Diabetes', 'Diabetes'],
        yticklabels=['No Diabetes', 'Diabetes'],
        linewidths=1,
        linecolor='white',
        annot_kws={'size': 14, 'weight': 'bold'},
        ax=ax
    )

    ax.set_title(f'Confusion Matrix — {model_name}',
                 fontsize=14, fontweight='bold', pad=16)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig