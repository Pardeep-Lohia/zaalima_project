import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .utils import setup_logging, ensure_directory, get_project_root
from .data_preprocessing import load_and_preprocess_data
from .feature_engineering import create_feature_pipeline

logger = setup_logging()

class ModelExplainer:
    """
    SHAP-based explainability for XGBoost/LightGBM production models.
    Provides global and local explanations to increase trust for maintenance engineers.
    """

    def __init__(self, model_path: str, feature_pipeline_path: str = 'models/feature_pipeline.joblib'):
        """
        Initialize the model explainer.

        Args:
            model_path: Path to the saved production model
            feature_pipeline_path: Path to the saved feature engineering pipeline
        """
        self.model_path = model_path
        self.feature_pipeline_path = feature_pipeline_path
        self.model = None
        self.feature_pipeline = None
        self.explainer = None
        self.feature_columns = None

        self._load_model_and_pipeline()

    def _load_model_and_pipeline(self) -> None:
        """Load the model and feature engineering pipeline."""
        try:
            # Load model
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            model_type = model_data['model_type']

            logger.info(f"Loaded {model_type} model from {self.model_path}")

            # Load feature pipeline
            self.feature_pipeline = joblib.load(self.feature_pipeline_path)
            logger.info(f"Loaded feature engineering pipeline from {self.feature_pipeline_path}")

            # Initialize SHAP explainer
            if model_type == 'xgboost':
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type == 'lightgbm':
                self.explainer = shap.TreeExplainer(self.model)
            else:
                raise ValueError(f"Unsupported model type for SHAP: {model_type}")

            logger.info("Initialized SHAP TreeExplainer")

        except Exception as e:
            logger.error(f"Failed to load model or pipeline: {e}")
            raise

    def explain_instance(self, X_instance: pd.DataFrame, save_path: str = None) -> Dict[str, Any]:
        """
        Generate local explanation (waterfall plot) for a single prediction.

        Args:
            X_instance: Single instance DataFrame with feature columns
            save_path: Optional path to save the waterfall plot

        Returns:
            Dictionary with SHAP values and explanation data
        """
        logger.info("Generating local explanation for single instance")

        # Ensure X_instance has correct columns and order
        X_instance = X_instance[self.feature_columns]

        # Get SHAP values
        shap_values = self.explainer.shap_values(X_instance)

        # For binary classification, shap_values might be 2D for both classes
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Take SHAP values for positive class (failure)
            shap_values = shap_values[1]

        # Generate waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            self.explainer.expected_value if not isinstance(self.explainer.expected_value, np.ndarray)
            else self.explainer.expected_value[1],  # For binary classification
            shap_values[0],  # First (and only) instance
            X_instance.iloc[0],
            show=False
        )
        plt.title("SHAP Waterfall Plot - Single Prediction Explanation")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved waterfall plot to {save_path}")

        plt.show()

        return {
            'shap_values': shap_values[0],
            'expected_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value,
            'feature_values': X_instance.iloc[0].to_dict()
        }

    def generate_global_report(self, X_sample: pd.DataFrame, save_path: str = None) -> Dict[str, Any]:
        """
        Generate global explanation (summary plot) for model understanding.

        Args:
            X_sample: Sample of validation data for global explanation
            save_path: Optional path to save the summary plot

        Returns:
            Dictionary with global SHAP statistics
        """
        logger.info("Generating global explanation report")

        # Ensure X_sample has correct columns and order
        X_sample = X_sample[self.feature_columns]

        # Sample a subset for efficiency (SHAP can be slow on large datasets)
        if len(X_sample) > 1000:
            X_sample = X_sample.sample(n=1000, random_state=42)
            logger.info("Sampled 1000 instances for global explanation")

        # Get SHAP values
        shap_values = self.explainer.shap_values(X_sample)

        # For binary classification, take positive class values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        # Generate summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            show=False,
            max_display=20  # Show top 20 features
        )
        plt.title("SHAP Summary Plot - Global Feature Importance")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary plot to {save_path}")

        plt.show()

        # Calculate feature importance statistics
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        return {
            'feature_importance': feature_importance,
            'mean_abs_shap': mean_abs_shap,
            'shap_values_sample': shap_values[:100] if len(shap_values) > 100 else shap_values  # Sample for storage
        }

def create_explainability_report(
    model_path: str = 'models/production_xgboost.joblib',
    feature_pipeline_path: str = 'models/feature_pipeline.joblib',
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Create complete explainability report using validation data.

    Args:
        model_path: Path to production model
        feature_pipeline_path: Path to feature pipeline
        save_plots: Whether to save plots to reports/

    Returns:
        Dictionary with explanation results
    """
    logger.info("Creating explainability report")

    # Initialize explainer
    explainer = ModelExplainer(model_path, feature_pipeline_path)

    # Load validation data
    _, val_df = load_and_preprocess_data()
    _, val_features = create_feature_pipeline(None, val_df, feature_pipeline_path)

    # Prepare validation features
    feature_cols = [col for col in val_features.columns if col not in ['timestamp', 'failure', 'failure_24h']]
    X_val = val_features[feature_cols]

    # Create reports directory
    reports_dir = Path(get_project_root()) / 'reports'
    ensure_directory(reports_dir)

    # Generate global report
    global_report = explainer.generate_global_report(
        X_val,
        save_path=str(reports_dir / 'shap_summary_plot.png') if save_plots else None
    )

    # Generate local explanation for a random failure instance
    failure_indices = val_features[val_features['failure_24h'] == 1].index
    if len(failure_indices) > 0:
        random_failure_idx = np.random.choice(failure_indices)
        X_failure = X_val.loc[[random_failure_idx]]

        local_explanation = explainer.explain_instance(
            X_failure,
            save_path=str(reports_dir / 'shap_waterfall_example.png') if save_plots else None
        )
    else:
        logger.warning("No failure instances found in validation data for local explanation")
        local_explanation = None

    results = {
        'global_explanation': global_report,
        'local_explanation': local_explanation,
        'plots_saved': save_plots,
        'validation_samples': len(X_val),
        'failure_samples': len(failure_indices) if len(failure_indices) > 0 else 0
    }

    logger.info("Explainability report completed")
    return results

if __name__ == "__main__":
    # Create explainability report
    results = create_explainability_report(save_plots=True)

    print("SHAP Explainability Report Generated:")
    print(f"- Validation samples: {results['validation_samples']}")
    print(f"- Failure samples: {results['failure_samples']}")
    print("- Global feature importance computed")
    if results['local_explanation']:
        print("- Local explanation (waterfall plot) generated")
    print("\nPlots saved to reports/ directory")
