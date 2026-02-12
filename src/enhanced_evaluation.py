import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    classification_report, roc_curve, roc_auc_score
)
import logging

from .utils import setup_logging, ensure_directory, get_project_root

logger = setup_logging()

class RareEventEvaluator:
    """
    Enhanced evaluation class for rare event detection in imbalanced datasets.
    Provides comprehensive metrics for predictive maintenance scenarios.
    """

    def __init__(self, y_true: pd.Series, y_pred_proba: np.ndarray, total_samples: int):
        """
        Initialize the rare event evaluator.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            total_samples: Total number of samples in the dataset
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.total_samples = total_samples

        # Calculate basic metrics
        self.precision, self.recall, self.thresholds = precision_recall_curve(y_true, y_pred_proba)
        self.pr_auc = auc(self.recall, self.precision)

        # Remove the last threshold (always 1.0) for cleaner analysis
        self.thresholds = self.thresholds[:-1]

        logger.info(f"Initialized RareEventEvaluator with {len(y_true)} samples, {y_true.sum()} positives")

    def recall_at_fixed_fpr(self, target_fpr: float = 0.05) -> Dict[str, Any]:
        """
        Calculate recall at a fixed false positive rate.

        Args:
            target_fpr: Target false positive rate (e.g., 0.05 for 5%)

        Returns:
            Dictionary with threshold, recall, precision, and other metrics
        """
        # Calculate FPR and TPR for each threshold
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_pred_proba)

        # Find threshold closest to target FPR
        fpr_diffs = np.abs(fpr - target_fpr)
        best_idx = np.argmin(fpr_diffs)
        threshold = roc_thresholds[best_idx]

        # Calculate metrics at this threshold
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr_actual = fp / (fp + tn) if (fp + tn) > 0 else 0

        results = {
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'fpr': fpr_actual,
            'target_fpr': target_fpr,
            'true_positives': tp,
            'false_positives': fp,
            'alert_rate': (tp + fp) / len(self.y_true)
        }

        logger.info(f"Recall at {target_fpr:.1%} FPR: {recall:.4f} (threshold: {threshold:.4f})")
        return results

    def precision_at_top_k(self, k: int) -> Dict[str, Any]:
        """
        Calculate precision when considering only top-K highest probability predictions as alerts.

        Args:
            k: Number of top predictions to consider

        Returns:
            Dictionary with precision, recall, and other metrics
        """
        # Get indices of top-K highest probabilities
        top_k_indices = np.argsort(self.y_pred_proba)[-k:]

        # Create predictions: only top-K are positive
        y_pred = np.zeros_like(self.y_pred_proba)
        y_pred[top_k_indices] = 1

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        threshold = self.y_pred_proba[top_k_indices[0]]  # Minimum threshold for top-K

        results = {
            'k': k,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'true_positives': tp,
            'false_positives': fp,
            'alert_rate': k / len(self.y_true)
        }

        logger.info(f"Precision @ top-{k}: {precision:.4f} (recall: {recall:.4f})")
        return results

    def simulate_daily_alert_volume(self, hours_per_day: int = 24) -> pd.DataFrame:
        """
        Simulate daily alert volume for different thresholds.

        Args:
            hours_per_day: Number of hours in a day (for scaling)

        Returns:
            DataFrame with daily alert volumes for different thresholds
        """
        # Sample thresholds to evaluate
        thresholds_to_test = np.linspace(0.01, 0.99, 50)

        results = []
        for threshold in thresholds_to_test:
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            alerts_per_hour = y_pred.sum() / len(self.y_pred_proba)
            alerts_per_day = alerts_per_hour * hours_per_day

            # Calculate precision and recall at this threshold
            if y_pred.sum() > 0:
                tp = ((y_pred == 1) & (self.y_true == 1)).sum()
                fp = ((y_pred == 1) & (self.y_true == 0)).sum()
                precision = tp / (tp + fp)
            else:
                precision = 0

            results.append({
                'threshold': threshold,
                'alerts_per_day': alerts_per_day,
                'precision': precision,
                'alert_rate': alerts_per_hour
            })

        return pd.DataFrame(results)

    def generate_evaluation_report(self, model_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for rare event detection.

        Args:
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing all evaluation metrics
        """
        # Basic metrics
        basic_metrics = {
            'pr_auc': self.pr_auc,
            'roc_auc': roc_auc_score(self.y_true, self.y_pred_proba),
            'avg_precision': np.mean(self.precision),
            'avg_recall': np.mean(self.recall)
        }

        # Recall at fixed FPRs
        fpr_targets = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
        recall_at_fpr = {}
        for fpr in fpr_targets:
            recall_at_fpr[f'fpr_{int(fpr*100)}'] = self.recall_at_fixed_fpr(fpr)

        # Precision at top-K
        k_values = [10, 25, 50, 100]  # Top-K alerts
        precision_at_k = {}
        for k in k_values:
            if k < len(self.y_true):  # Only if K is reasonable
                precision_at_k[f'top_{k}'] = self.precision_at_top_k(k)

        # Daily alert volume simulation
        daily_alerts = self.simulate_daily_alert_volume()
        # Align array lengths safely (required for rare-event data)
        min_len = min(
            len(self.thresholds),
            len(self.precision) - 1,
            len(self.recall) - 1
        )
        # Threshold vs metrics table
        threshold_table = pd.DataFrame({
            'threshold': self.thresholds[:min_len],
            'precision': self.precision[:min_len],
            'recall': self.recall[:min_len]
        })
        threshold_table['f1'] = (
        2 * threshold_table['precision'] * threshold_table['recall'] /
        (threshold_table['precision'] + threshold_table['recall'] + 1e-9)
        )

        threshold_table = threshold_table.fillna(0)

        report = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'recall_at_fpr': recall_at_fpr,
            'precision_at_k': precision_at_k,
            'daily_alert_simulation': daily_alerts,
            'threshold_vs_metrics': threshold_table,
            'rare_event_metrics': []  # Will be populated below
        }

        # Create summary metrics for rare events
        rare_event_metrics = []

        # Add recall at FPR metrics
        for key, metrics in recall_at_fpr.items():
            rare_event_metrics.append({
                'metric_type': 'recall_at_fpr',
                'metric_name': f'Recall @ {metrics["target_fpr"]:.1%} FPR',
                'value': metrics['recall'],
                'threshold': metrics['threshold'],
                'additional_info': f'Precision: {metrics["precision"]:.3f}, Alerts/day: {metrics["alert_rate"]*24:.1f}'
            })

        # Add precision at K metrics
        for key, metrics in precision_at_k.items():
            rare_event_metrics.append({
                'metric_type': 'precision_at_k',
                'metric_name': f'Precision @ Top-{metrics["k"]}',
                'value': metrics['precision'],
                'threshold': metrics['threshold'],
                'additional_info': f'Recall: {metrics["recall"]:.3f}, Alerts/day: {metrics["alert_rate"]*24:.1f}'
            })

        report['rare_event_metrics'] = rare_event_metrics

        logger.info(f"Generated evaluation report for {model_name}")
        return report

    def get_business_interpretation(self, threshold_metrics: Optional[Dict] = None) -> str:
        """
        Generate business interpretation of rare event detection results.

        Args:
            threshold_metrics: Optional precision-driven threshold metrics

        Returns:
            String with business interpretation
        """
        interpretation = []

        # Overall performance summary
        interpretation.append("## Business Impact Assessment")
        interpretation.append(f"- **PR-AUC**: {self.pr_auc:.3f} (higher is better for rare events)")
        interpretation.append(f"- **Failure Detection Rate**: {self.y_true.sum()}/{len(self.y_true)} ({self.y_true.mean():.2%})")

        # Threshold-based metrics
        if threshold_metrics:
            alerts_per_day = threshold_metrics['alert_rate'] * 24
            fp_per_1000 = (threshold_metrics['false_positives'] / len(self.y_true)) * 1000

            interpretation.append("\n## Operational Metrics (at 60% Precision Threshold)")
            interpretation.append(f"- **Precision**: {threshold_metrics['precision']:.1%} (alert quality)")
            interpretation.append(f"- **Recall**: {threshold_metrics['recall']:.1%} (failure detection rate)")
            interpretation.append(f"- **Alerts per Day**: {alerts_per_day:.1f}")
            interpretation.append(f"- **False Positives per 1000 Predictions**: {fp_per_1000:.1f}")

            # Business recommendations
            interpretation.append("\n## Recommendations")
            if threshold_metrics['precision'] >= 0.6:
                interpretation.append("✓ **Threshold meets precision requirement** (>=60%)")
            else:
                interpretation.append("⚠ **Threshold below precision target** - consider higher threshold")

            if alerts_per_day <= 10:
                interpretation.append("✓ **Alert volume manageable** (<=10/day)")
            else:
                interpretation.append("⚠ **High alert volume** - may overwhelm maintenance team")

            if fp_per_1000 <= 5:
                interpretation.append("✓ **Low false positive rate** - good alert quality")
            else:
                interpretation.append("⚠ **High false positive rate** - too many false alarms")

        return "\n".join(interpretation)

    def plot_comprehensive_evaluation(self, model_name: str, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive evaluation plots for rare event detection.

        Args:
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Rare Event Evaluation: {model_name}', fontsize=16)

        # Plot 1: Precision-Recall Curve
        axes[0, 0].plot(self.recall, self.precision, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title(f'PR Curve (AUC = {self.pr_auc:.4f})')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Threshold vs Precision/Recall
        ax2 = axes[0, 1]
        min_len = min(
            len(self.thresholds),
            len(self.precision),
            len(self.recall)
        )
        
        thresholds = self.thresholds[:min_len]
        precision = self.precision[:min_len]
        recall = self.recall[:min_len]
        
        ax2.plot(thresholds, precision, label='Precision', linewidth=2)
        ax2.plot(thresholds, recall, label='Recall', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Threshold vs Precision/Recall')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Daily Alert Volume Simulation
        daily_alerts = self.simulate_daily_alert_volume()
        axes[1, 0].plot(daily_alerts['threshold'], daily_alerts['alerts_per_day'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Alerts per Day')
        axes[1, 0].set_title('Daily Alert Volume vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Precision vs Alert Rate
        axes[1, 1].scatter(daily_alerts['alert_rate'] * 24, daily_alerts['precision'],
                          c=daily_alerts['threshold'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel('Alerts per Day')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Alert Volume')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comprehensive evaluation plot to {save_path}")
        else:
            plt.show()

def compare_models_rare_events(model_results: Dict[str, Dict], total_samples: int) -> pd.DataFrame:
    """
    Compare multiple models on rare event metrics.

    Args:
        model_results: Dictionary of model results
        total_samples: Total number of samples

    Returns:
        DataFrame with model comparison
    """
    comparison_data = []

    for model_name, results in model_results.items():
        val_results = results.get('val_results', {})

        row = {
            'model': model_name,
            'pr_auc': val_results.get('pr_auc', 0),
            'accuracy': val_results.get('accuracy', 0),
            'failure_precision': val_results.get('failure_precision', 0),
            'failure_recall': val_results.get('failure_recall', 0),
            'failure_f1': val_results.get('failure_f1', 0)
        }

        # Add threshold optimization results if available
        threshold_opt = results.get('threshold_optimization', {})
        if threshold_opt:
            # Add best threshold metrics
            best_threshold = threshold_opt.get('threshold_table', pd.DataFrame())
            if not best_threshold.empty:
                # Find threshold with best F1
                best_f1_idx = best_threshold['f1'].idxmax()
                row.update({
                    'best_threshold': best_threshold.loc[best_f1_idx, 'threshold'],
                    'best_precision': best_threshold.loc[best_f1_idx, 'precision'],
                    'best_recall': best_threshold.loc[best_f1_idx, 'recall'],
                    'best_f1': best_threshold.loc[best_f1_idx, 'f1']
                })

        # Add precision-driven threshold metrics
        precision_opt = results.get('precision_threshold', {})
        if precision_opt:
            row.update({
                'precision_threshold': precision_opt.get('threshold', 0),
                'precision_at_threshold': precision_opt.get('precision', 0),
                'recall_at_threshold': precision_opt.get('recall', 0),
                'alerts_per_day': precision_opt.get('alert_rate', 0) * 24,
                'false_positives_per_1000': (precision_opt.get('false_positives', 0) / total_samples) * 1000
            })

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('pr_auc', ascending=False)

    logger.info("Generated model comparison for rare event detection")
    return comparison_df

def create_executive_summary(model_comparison: pd.DataFrame, best_model_results: Dict) -> str:
    """
    Create an executive summary explaining results and recommendations.

    Args:
        model_comparison: DataFrame with model comparison
        best_model_results: Results for the best performing model

    Returns:
        Markdown string with executive summary
    """
    best_model = model_comparison.iloc[0]['model']
    best_pr_auc = model_comparison.iloc[0]['pr_auc']

    summary = f"""# Predictive Maintenance Model Evaluation Summary

## Executive Summary

This report evaluates predictive maintenance models for detecting equipment failures 24 hours in advance using IoT sensor data (temperature, vibration, pressure).

## Key Findings

### Best Performing Model
- **Model**: {best_model}
- **PR-AUC**: {best_pr_auc:.4f}
- **Failure Recall**: {model_comparison.iloc[0]['failure_recall']:.4f}
- **Failure Precision**: {model_comparison.iloc[0]['failure_precision']:.4f}

### Model Performance Comparison
{model_comparison.to_markdown(index=False)}

## Recommendations

### Threshold Selection Strategy
Based on the evaluation, we recommend using a threshold that balances recall and alert volume:

1. **High Recall Scenario (90% target)**: Use threshold ≈ 0.1-0.2
   - Captures most failures but generates more alerts
   - Suitable for critical systems where missing failures is costly

2. **Balanced Scenario (70-80% recall)**: Use threshold ≈ 0.3-0.5
   - Good balance of failure detection and alert volume
   - Recommended for most production environments

3. **Low Alert Volume Scenario**: Use threshold > 0.5
   - Minimizes false alarms but may miss some failures
   - Suitable when maintenance resources are limited

### Business Impact
- **Current Baseline**: Near-random performance (PR-AUC ≈ 0.5)
- **Improvement**: {best_pr_auc - 0.5:.3f} increase in PR-AUC
- **Failure Detection**: {model_comparison.iloc[0]['failure_recall']:.1%} of failures can be predicted 24 hours in advance

## Technical Details

### Data Characteristics
- Total samples: 10,000 (hourly readings)
- Failure rate: ~0.5-0.6%
- Features: 135 engineered time-series features
- Time-aware train/validation split to prevent data leakage

### Model Architecture
- XGBoost/LightGBM with scale_pos_weight for class imbalance
- Optuna hyperparameter optimization (50 trials)
- PR-AUC as primary evaluation metric

### Evaluation Metrics
- **PR-AUC**: Primary metric for imbalanced classification
- **Recall @ Fixed FPR**: Measures failure detection at acceptable false positive rates
- **Precision @ Top-K**: Evaluates alert quality when limiting daily alerts
- **Daily Alert Simulation**: Predicts operational alert volume

## Next Steps

1. **Deployment Preparation**
   - Implement model in production pipeline
   - Set up monitoring for model performance drift
   - Establish alert threshold based on business requirements

2. **Further Improvements**
   - Consider ensemble methods for better performance
   - Implement online learning for concept drift
   - Add more sensor types if available

3. **Operational Integration**
   - Define maintenance response workflows
   - Train maintenance teams on alert interpretation
   - Establish feedback loop for model improvement

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return summary
