import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

from .utils import setup_logging, ensure_directory, get_project_root

logger = setup_logging()

class ThresholdOptimizer:
    """
    Optimize decision thresholds for imbalanced classification.
    Supports fixed recall targets, alert budget constraints, and cost-sensitive optimization.
    """

    def __init__(self, y_true: pd.Series, y_pred_proba: np.ndarray):
        """
        Initialize threshold optimizer.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.precision, self.recall, self.thresholds = precision_recall_curve(y_true, y_pred_proba)

        # Ensure all arrays have the same length (remove last threshold which is always 1.0)
        min_len = min(len(self.precision), len(self.recall), len(self.thresholds) - 1)
        self.precision = self.precision[:min_len]
        self.recall = self.recall[:min_len]
        self.thresholds = self.thresholds[:min_len]

    def find_threshold_for_precision(self, min_precision: float = 0.6) -> Dict[str, Any]:
        """
        Find threshold that achieves at least minimum precision.

        Args:
            min_precision: Minimum precision level (0-1)

        Returns:
            Dictionary with threshold, precision, recall, and other metrics
        """
        # Find thresholds that achieve at least minimum precision
        valid_indices = self.precision >= min_precision
        if not valid_indices.any():
            logger.warning(f"Cannot achieve precision >= {min_precision}. Using best available.")
            best_idx = np.argmax(self.precision)
        else:
            # Among thresholds that meet precision, choose the one with highest recall
            valid_recalls = self.recall[valid_indices]
            valid_indices_array = np.where(valid_indices)[0]
            best_idx = valid_indices_array[np.argmax(valid_recalls)]

        threshold = self.thresholds[best_idx]
        precision = self.precision[best_idx]
        recall = self.recall[best_idx]

        # Calculate predictions at this threshold
        y_pred = (self.y_pred_proba >= threshold).astype(int)

        # Calculate additional metrics
        cm = confusion_matrix(self.y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'alert_rate': (tp + fp) / len(self.y_true),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        }

        logger.info(f"Threshold for {min_precision:.0%} precision: {threshold:.4f}")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.info(f"  Alert rate: {results['alert_rate']:.4f}")

        return results

    def get_precision_threshold_for_evaluation(self, min_precision: float = 0.6) -> Dict[str, Any]:
        """
        Get precision-driven threshold results for evaluation pipeline integration.

        Args:
            min_precision: Minimum precision level (0-1)

        Returns:
            Dictionary with threshold and metrics for evaluation
        """
        return self.find_threshold_for_precision(min_precision)

    def find_threshold_for_recall(self, target_recall: float) -> Dict[str, Any]:
        """
        Find threshold that achieves at least target recall.

        Args:
            target_recall: Target recall level (0-1)

        Returns:
            Dictionary with threshold, precision, recall, and other metrics
        """
        # Find the highest threshold that achieves at least target recall
        valid_indices = self.recall >= target_recall
        if not valid_indices.any():
            logger.warning(f"Cannot achieve recall >= {target_recall}. Using best available.")
            best_idx = np.argmax(self.recall)
        else:
            # Among thresholds that meet recall, choose the one with highest precision
            valid_thresholds = self.thresholds[valid_indices]
            valid_precisions = self.precision[valid_indices]
            best_idx = np.argmax(valid_precisions)

        threshold = self.thresholds[best_idx]
        precision = self.precision[best_idx]
        recall = self.recall[best_idx]

        # Calculate predictions at this threshold
        y_pred = (self.y_pred_proba >= threshold).astype(int)

        # Calculate additional metrics
        cm = confusion_matrix(self.y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'alert_rate': (tp + fp) / len(self.y_true),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        }

        logger.info(f"Threshold for {target_recall:.0%} recall: {threshold:.4f}")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.info(f"  Alert rate: {results['alert_rate']:.4f}")

        return results

    def find_threshold_for_alert_budget(self, max_alerts_per_day: int, total_samples: int) -> Dict[str, Any]:
        """
        Find threshold that limits alerts to specified daily budget.

        Args:
            max_alerts_per_day: Maximum number of alerts allowed per day
            total_samples: Total number of samples in dataset

        Returns:
            Dictionary with threshold and metrics
        """
        # Calculate maximum alert rate
        max_alert_rate = max_alerts_per_day / (total_samples / 24)  # Assuming hourly data

        # Find threshold that gives alert rate <= max_alert_rate
        # Sort by threshold ascending (higher threshold = fewer alerts)
        sorted_indices = np.argsort(self.thresholds)[::-1]  # Descending threshold

        best_threshold = None
        best_metrics = None

        for idx in sorted_indices:
            threshold = self.thresholds[idx]
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            alert_rate = y_pred.sum() / len(y_pred)

            if alert_rate <= max_alert_rate:
                precision = self.precision[idx]
                recall = self.recall[idx]

                cm = confusion_matrix(self.y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()

                metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
                    'true_positives': tp,
                    'false_positives': fp,
                    'alert_rate': alert_rate,
                    'alerts_per_day': alert_rate * (total_samples / 24),
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
                }

                if best_metrics is None or precision > best_metrics['precision']:
                    best_threshold = threshold
                    best_metrics = metrics

        if best_metrics is None:
            logger.warning(f"Cannot achieve {max_alerts_per_day} alerts/day. Using lowest threshold.")
            return self.find_threshold_for_recall(0.1)  # Fallback to 10% recall

        logger.info(f"Threshold for {max_alerts_per_day} alerts/day: {best_threshold:.4f}")
        logger.info(f"  Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
        logger.info(f"  Actual alerts/day: {best_metrics['alerts_per_day']:.1f}")

        return best_metrics

    def get_threshold_vs_metrics_table(self) -> pd.DataFrame:
        """
        Generate comprehensive table of threshold vs metrics.

        Returns:
            DataFrame with threshold, precision, recall, f1, alert_rate
        """
        results = []

        for i, threshold in enumerate(self.thresholds):
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            alert_rate = y_pred.sum() / len(y_pred)

            results.append({
                'threshold': threshold,
                'precision': self.precision[i],
                'recall': self.recall[i],
                'f1': 2 * self.precision[i] * self.recall[i] / (self.precision[i] + self.recall[i]) if (self.precision[i] + self.recall[i]) > 0 else 0,
                'alert_rate': alert_rate,
                'alerts_per_day': alert_rate * (len(self.y_true) / 24)  # Assuming hourly data
            })

        return pd.DataFrame(results)

    def plot_threshold_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plot threshold vs precision/recall/alert rate.

        Args:
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Precision and Recall vs Threshold
        ax1.plot(self.thresholds, self.precision, 'b-', label='Precision', linewidth=2)
        ax1.plot(self.thresholds, self.recall, 'r-', label='Recall', linewidth=2)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision and Recall vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Alert Rate vs Threshold
        alert_rates = []
        for threshold in self.thresholds:
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            alert_rates.append(y_pred.sum() / len(y_pred))

        ax2.plot(self.thresholds, alert_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Alert Rate')
        ax2.set_title('Alert Rate vs Threshold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            ensure_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved threshold analysis plot to {save_path}")

        plt.show()

    def plot_confusion_matrix_at_threshold(self, threshold: float, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix at specific threshold.

        Args:
            threshold: Decision threshold
            save_path: Optional path to save the plot
        """
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Failure'],
                   yticklabels=['Normal', 'Failure'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('.3f')

        if save_path:
            ensure_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.show()

def optimize_thresholds(y_true: pd.Series, y_pred_proba: np.ndarray,
                       total_samples: int = 10000) -> Dict[str, Any]:
    """
    Complete threshold optimization analysis.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        total_samples: Total samples in dataset for alert rate calculation

    Returns:
        Dictionary with optimization results
    """
    optimizer = ThresholdOptimizer(y_true, y_pred_proba)

    # Optimize for different recall targets
    recall_targets = [0.7, 0.8, 0.9]
    recall_results = {}
    for target in recall_targets:
        recall_results[f'recall_{int(target*100)}'] = optimizer.find_threshold_for_recall(target)

    # Optimize for alert budgets
    alert_budgets = [1, 5, 10]  # alerts per day
    budget_results = {}
    for budget in alert_budgets:
        budget_results[f'budget_{budget}'] = optimizer.find_threshold_for_alert_budget(budget, total_samples)

    # Get comprehensive threshold table
    threshold_table = optimizer.get_threshold_vs_metrics_table()

    results = {
        'recall_optimization': recall_results,
        'budget_optimization': budget_results,
        'threshold_table': threshold_table,
        'optimizer': optimizer
    }

    return results

def evaluate_rare_event_metrics(y_true: pd.Series, y_pred_proba: np.ndarray,
                               thresholds: List[float]) -> pd.DataFrame:
    """
    Evaluate metrics specifically designed for rare event detection.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        thresholds: List of thresholds to evaluate

    Returns:
        DataFrame with rare event metrics
    """
    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Standard metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Rare event specific metrics
        alert_rate = (tp + fp) / len(y_true)
        alerts_per_day = alert_rate * (len(y_true) / 24)  # Assuming hourly data

        # Precision at top-K (simulate daily top alerts)
        daily_samples = len(y_true) // 24
        top_k_precisions = []

        for day in range(min(24, len(y_true) // daily_samples)):
            start_idx = day * daily_samples
            end_idx = (day + 1) * daily_samples
            day_proba = y_pred_proba[start_idx:end_idx]
            day_true = y_true.iloc[start_idx:end_idx]

            # Get top alerts for the day
            n_alerts = min(5, len(day_proba))  # Top 5 alerts per day
            top_indices = np.argsort(day_proba)[::-1][:n_alerts]
            top_true = day_true.iloc[top_indices]

            if len(top_true) > 0:
                top_k_precisions.append(top_true.sum() / len(top_true))

        avg_top_k_precision = np.mean(top_k_precisions) if top_k_precisions else 0

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'alert_rate': alert_rate,
            'alerts_per_day': alerts_per_day,
            'top_5_precision': avg_top_k_precision,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        })

    return pd.DataFrame(results)
