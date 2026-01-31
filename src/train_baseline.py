import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .utils import setup_logging, ensure_directory, get_project_root
from .data_preprocessing import load_and_preprocess_data, analyze_class_distribution
from .feature_engineering import create_feature_pipeline

logger = setup_logging()

class BaselineModelTrainer:
    """
    Production-ready baseline model trainer for imbalanced classification.
    Supports Logistic Regression and Random Forest with class weights.
    """

    def __init__(self, model_type: str = 'random_forest', random_seed: int = 42):
        """
        Initialize the baseline trainer.

        Args:
            model_type: Type of baseline model ('logistic_regression' or 'random_forest')
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_seed = random_seed
        self.model = None
        self.feature_columns = None

        # Set random seeds
        np.random.seed(random_seed)

        logger.info(f"Initialized {model_type} baseline trainer with seed {random_seed}")

    def _create_model(self) -> None:
        """
        Create the baseline model with appropriate configuration for imbalanced data.
        """
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                class_weight='balanced',
                solver='liblinear',
                random_state=self.random_seed,
                max_iter=1000
            )
            logger.info("Created Logistic Regression model with balanced class weights")

        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                class_weight='balanced',
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_seed,
                n_jobs=-1
            )
            logger.info("Created Random Forest model with balanced class weights")

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the baseline model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_type} baseline model...")

        self.feature_columns = X_train.columns.tolist()
        self._create_model()
        self.model.fit(X_train, y_train)

        logger.info("Baseline model training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics.

        Args:
            X: Features for evaluation
            y_true: True labels
            dataset_name: Name of the dataset for logging

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {self.model_type} on {dataset_name}")

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        results = {
            'accuracy': accuracy,
            'pr_auc': pr_auc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': cm,
            'classification_report': report
        }

        logger.info(f"{dataset_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
        logger.info(f"  F1-Score: {results['f1_score']:.4f}")

        return results

    def plot_precision_recall_curve(self, X: pd.DataFrame, y_true: pd.Series, save_path: Optional[str] = None) -> None:
        """
        Plot Precision-Recall curve.

        Args:
            X: Features for evaluation
            y_true: True labels
            save_path: Optional path to save the plot
        """
        y_pred_proba = self.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_type.title().replace("_", " ")} Baseline')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            ensure_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, X: pd.DataFrame, y_true: pd.Series, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.

        Args:
            X: Features for evaluation
            y_true: True labels
            save_path: Optional path to save the plot
        """
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Failure'],
                   yticklabels=['Normal', 'Failure'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.model_type.title().replace("_", " ")} Baseline')

        if save_path:
            ensure_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.show()

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with feature importance scores, or None if not supported
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For Logistic Regression, use absolute coefficient values
            importance_scores = np.abs(self.model.coef_[0])
        else:
            logger.warning(f"Model {self.model_type} does not support feature importance")
            return None

        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        return feature_importance

    def save_model(self, path: str) -> None:
        """
        Save the trained model and metadata.

        Args:
            path: Path to save the model
        """
        ensure_directory(Path(path).parent)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'random_seed': self.random_seed
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved baseline model to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.random_seed = model_data['random_seed']
        logger.info(f"Loaded baseline model from {path}")

def train_baseline_models(save_models: bool = True, save_plots: bool = True) -> Dict[str, Any]:
    """
    Train and evaluate baseline models (Logistic Regression and Random Forest).

    Args:
        save_models: Whether to save trained models
        save_plots: Whether to save evaluation plots

    Returns:
        Dictionary with training results and comparisons
    """
    logger.info("Starting baseline model training pipeline")

    # Load and preprocess data
    train_df, val_df = load_and_preprocess_data()

    # Analyze class distribution
    analyze_class_distribution(train_df, "Training Set")
    analyze_class_distribution(val_df, "Validation Set")

    # Create feature engineering pipeline
    train_features, val_features = create_feature_pipeline(train_df, val_df)

    # Prepare features and labels
    feature_cols = [col for col in train_features.columns if col not in ['timestamp', 'failure', 'failure_24h']]
    X_train = train_features[feature_cols]
    y_train = train_features['failure_24h']
    X_val = val_features[feature_cols]
    y_val = val_features['failure_24h']

    logger.info(f"Training with {X_train.shape[1]} features")

    # Train Logistic Regression
    lr_trainer = BaselineModelTrainer('logistic_regression')
    lr_trainer.train(X_train, y_train)

    # Train Random Forest
    rf_trainer = BaselineModelTrainer('random_forest')
    rf_trainer.train(X_train, y_train)

    # Evaluate both models
    lr_train_results = lr_trainer.evaluate(X_train, y_train, "Training Set")
    lr_val_results = lr_trainer.evaluate(X_val, y_val, "Validation Set")

    rf_train_results = rf_trainer.evaluate(X_train, y_train, "Training Set")
    rf_val_results = rf_trainer.evaluate(X_val, y_val, "Validation Set")

    # Get feature importance
    lr_feature_importance = lr_trainer.get_feature_importance()
    rf_feature_importance = rf_trainer.get_feature_importance()

    # Save models if requested
    if save_models:
        models_dir = Path(get_project_root()) / 'models'
        ensure_directory(models_dir)
        lr_trainer.save_model(str(models_dir / 'baseline_logistic_regression.joblib'))
        rf_trainer.save_model(str(models_dir / 'baseline_random_forest.joblib'))

    # Create reports directory
    reports_dir = Path(get_project_root()) / 'reports'
    ensure_directory(reports_dir)

    # Save plots if requested
    if save_plots:
        lr_trainer.plot_precision_recall_curve(
            X_val, y_val,
            save_path=str(reports_dir / 'baseline_lr_pr_curve.png')
        )
        lr_trainer.plot_confusion_matrix(
            X_val, y_val,
            save_path=str(reports_dir / 'baseline_lr_confusion_matrix.png')
        )

        rf_trainer.plot_precision_recall_curve(
            X_val, y_val,
            save_path=str(reports_dir / 'baseline_rf_pr_curve.png')
        )
        rf_trainer.plot_confusion_matrix(
            X_val, y_val,
            save_path=str(reports_dir / 'baseline_rf_confusion_matrix.png')
        )

    # Save feature importance
    if lr_feature_importance is not None:
        lr_feature_importance.to_csv(reports_dir / 'baseline_lr_feature_importance.csv', index=False)
    if rf_feature_importance is not None:
        rf_feature_importance.to_csv(reports_dir / 'baseline_rf_feature_importance.csv', index=False)

    # Compare models
    lr_pr_auc = lr_val_results['pr_auc']
    rf_pr_auc = rf_val_results['pr_auc']

    if rf_pr_auc > lr_pr_auc:
        best_model = 'random_forest'
        best_pr_auc = rf_pr_auc
        best_trainer = rf_trainer
    else:
        best_model = 'logistic_regression'
        best_pr_auc = lr_pr_auc
        best_trainer = lr_trainer

    logger.info("Baseline Model Comparison:")
    logger.info(f"  Logistic Regression PR-AUC: {lr_pr_auc:.4f}")
    logger.info(f"  Random Forest PR-AUC: {rf_pr_auc:.4f}")
    logger.info(f"  Best baseline model: {best_model} (PR-AUC: {best_pr_auc:.4f})")

    results = {
        'logistic_regression': {
            'train_results': lr_train_results,
            'val_results': lr_val_results,
            'feature_importance': lr_feature_importance
        },
        'random_forest': {
            'train_results': rf_train_results,
            'val_results': rf_val_results,
            'feature_importance': rf_feature_importance
        },
        'best_model': best_model,
        'best_pr_auc': best_pr_auc,
        'comparison': {
            'lr_pr_auc': lr_pr_auc,
            'rf_pr_auc': rf_pr_auc
        }
    }

    logger.info("Baseline model training pipeline completed")

    return results

if __name__ == "__main__":
    # Train baseline models
    baseline_results = train_baseline_models()

    print("\n" + "="*50)
    print("BASELINE MODEL RESULTS")
    print("="*50)
    print(f"Logistic Regression PR-AUC: {baseline_results['comparison']['lr_pr_auc']:.4f}")
    print(f"Random Forest PR-AUC: {baseline_results['comparison']['rf_pr_auc']:.4f}")
    print(f"Best baseline model: {baseline_results['best_model']} (PR-AUC: {baseline_results['best_pr_auc']:.4f})")
