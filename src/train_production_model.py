import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import warnings

from .utils import setup_logging, ensure_directory, get_project_root
from .data_preprocessing import load_and_preprocess_data, analyze_class_distribution
from .feature_engineering import create_feature_pipeline
from .threshold_optimization import ThresholdOptimizer
from .enhanced_evaluation import RareEventEvaluator

logger = setup_logging()
warnings.filterwarnings('ignore')

class ProductionModelTrainer:
    """
    Production-ready model trainer with hyperparameter tuning for imbalanced classification.
    """

    def __init__(self, model_type: str = 'xgboost', random_seed: int = 42):
        """
        Initialize the production trainer.

        Args:
            model_type: Type of production model ('xgboost' or 'lightgbm')
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_seed = random_seed
        self.model = None
        self.feature_columns = None
        self.best_params = None

        # Set random seeds
        np.random.seed(random_seed)

    def _calculate_class_weights(self, y_train: pd.Series) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced dataset.

        Args:
            y_train: Training labels

        Returns:
            Dictionary with class weights
        """
        n_samples = len(y_train)
        n_classes = y_train.nunique()
        class_counts = y_train.value_counts()

        # Calculate weights inversely proportional to class frequency
        weights = {}
        for class_label, count in class_counts.items():
            weights[class_label] = n_samples / (n_classes * count)

        logger.info(f"Calculated class weights: {weights}")
        return weights

    def _objective_xgboost(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Optuna objective function for XGBoost hyperparameter tuning.

        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training labels

        Returns:
            PR-AUC score for the trial
        """
        class_weights = self._calculate_class_weights(y_train)
        base_scale_pos_weight = class_weights[1] / class_weights[0]

        # Define hyperparameter search space including scale_pos_weight
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',  # Precision-Recall AUC
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1 * base_scale_pos_weight, 10 * base_scale_pos_weight, log=True),
            'random_state': self.random_seed,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }

        # Train model with cross-validation
        model = xgb.XGBClassifier(**params)

        # Use PR-AUC as scoring metric
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='average_precision')
        return scores.mean()

    def _objective_lightgbm(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Optuna objective function for LightGBM hyperparameter tuning.

        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training labels

        Returns:
            PR-AUC score for the trial
        """
        class_weights = self._calculate_class_weights(y_train)

        params = {
            'objective': 'binary',
            'metric': 'average_precision',  # PR-AUC
            'class_weight': class_weights,
            'random_state': self.random_seed,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        }

        model = lgb.LGBMClassifier(**params, verbosity=-1)

        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='average_precision')
        return scores.mean()

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials

        Returns:
            Best hyperparameters found
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type} with {n_trials} trials")

        if self.model_type == 'xgboost':
            objective_func = self._objective_xgboost
        elif self.model_type == 'lightgbm':
            objective_func = self._objective_lightgbm
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Create Optuna study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(lambda trial: objective_func(trial, X_train, y_train), n_trials=n_trials)

        self.best_params = study.best_params
        logger.info(f"Best hyperparameters found: {self.best_params}")
        logger.info(f"Best CV PR-AUC: {study.best_value:.4f}")

        return self.best_params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, tune_params: bool = True) -> None:
        """
        Train the production model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels
            tune_params: Whether to perform hyperparameter tuning
        """
        logger.info(f"Training {self.model_type} production model...")

        self.feature_columns = X_train.columns.tolist()

        if tune_params:
            # Perform hyperparameter tuning
            best_params = self.tune_hyperparameters(X_train, y_train)
        else:
            # Use default parameters
            best_params = {}

        # Create final model with best parameters
        class_weights = self._calculate_class_weights(y_train)

        if self.model_type == 'xgboost':
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'scale_pos_weight': class_weights[1] / class_weights[0],
                'random_state': self.random_seed,
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
            default_params.update(best_params)
            self.model = xgb.XGBClassifier(**default_params)

        elif self.model_type == 'lightgbm':
            default_params = {
                'objective': 'binary',
                'metric': 'average_precision',
                'class_weight': class_weights,
                'random_state': self.random_seed,
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
            default_params.update(best_params)
            self.model = lgb.LGBMClassifier(**default_params, verbosity=-1)

        # Train final model
        if self.model_type == 'xgboost':
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        logger.info("Production model training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics.
        """
        logger.info(f"Evaluating production model on {dataset_name}")

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        # Failure-class specific metrics (robust against missing class)
        failure_metrics = report.get('1', {'precision': 0, 'recall': 0, 'f1-score': 0})
        failure_precision = failure_metrics['precision']
        failure_recall = failure_metrics['recall']
        failure_f1 = failure_metrics['f1-score']

        results = {
            'accuracy': accuracy,
            'pr_auc': pr_auc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'failure_precision': failure_precision,
            'failure_recall': failure_recall,
            'failure_f1': failure_f1,
            'confusion_matrix': cm,
            'classification_report': report
        }

        logger.info(f"{dataset_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        logger.info(f"  Failure-Class Precision: {failure_precision:.4f}")
        logger.info(f"  Failure-Class Recall: {failure_recall:.4f}")
        logger.info(f"  Failure-Class F1: {failure_f1:.4f}")

        return results

    def plot_precision_recall_curve(self, X: pd.DataFrame, y_true: pd.Series, save_path: str = None) -> None:
        """Plot Precision-Recall curve."""
        y_pred_proba = self.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_type.title()} Production Model')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, X: pd.DataFrame, y_true: pd.Series, save_path: str = None) -> None:
        """Plot confusion matrix."""
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Failure'],
                   yticklabels=['Normal', 'Failure'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.model_type.title()} Production Model')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.show()

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        else:
            logger.warning("Model does not support feature importance")
            return None

        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        return feature_importance

    def save_model(self, path: str) -> None:
        """Save the trained model and metadata."""
        ensure_directory(Path(path).parent)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params,
            'random_seed': self.random_seed
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved production model to {path}")

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.best_params = model_data['best_params']
        self.random_seed = model_data['random_seed']
        logger.info(f"Loaded production model from {path}")

def train_production_model(
    model_type: str = 'xgboost',
    tune_hyperparams: bool = True,
    n_trials: int = 50,
    save_model: bool = True,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline to train and evaluate production model.

    Args:
        model_type: Type of production model
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_trials: Number of Optuna trials
        save_model: Whether to save the trained model
        save_plots: Whether to save evaluation plots

    Returns:
        Dictionary with training results and metrics
    """
    logger.info("Starting production model training pipeline")

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

    # Train production model
    trainer = ProductionModelTrainer(model_type=model_type)
    trainer.train(X_train, y_train, tune_params=tune_hyperparams)

    # Evaluate on training set
    train_results = trainer.evaluate(X_train, y_train, "Training Set")

    # Evaluate on validation set
    val_results = trainer.evaluate(X_val, y_val, "Validation Set")

    # Get feature importance
    feature_importance = trainer.get_feature_importance()

    # Save model if requested
    if save_model:
        model_path = Path(get_project_root()) / 'models' / f'production_{model_type}.joblib'
        trainer.save_model(str(model_path))

    # Create reports directory
    reports_dir = Path(get_project_root()) / 'reports'
    ensure_directory(reports_dir)

    # Save plots if requested
    if save_plots:
        trainer.plot_precision_recall_curve(
            X_val, y_val,
            save_path=str(reports_dir / f'production_{model_type}_pr_curve.png')
        )
        trainer.plot_confusion_matrix(
            X_val, y_val,
            save_path=str(reports_dir / f'production_{model_type}_confusion_matrix.png')
        )

    # Save feature importance
    if feature_importance is not None:
        feature_importance.to_csv(reports_dir / f'production_{model_type}_feature_importance.csv', index=False)

    # Save best hyperparameters
    if trainer.best_params:
        pd.DataFrame([trainer.best_params]).to_csv(
            reports_dir / f'production_{model_type}_best_params.csv', index=False
        )

    # Perform threshold optimization for precision-driven threshold
    threshold_optimizer = ThresholdOptimizer(y_val, trainer.predict_proba(X_val)[:, 1])
    precision_threshold = threshold_optimizer.get_precision_threshold_for_evaluation(min_precision=0.6)

    # Create rare event evaluator for business interpretation
    rare_evaluator = RareEventEvaluator(y_val, trainer.predict_proba(X_val)[:, 1], len(y_val))
    business_interpretation = rare_evaluator.get_business_interpretation(threshold_metrics=precision_threshold)

    results = {
        'model_type': model_type,
        'tuned': tune_hyperparams,
        'train_results': train_results,
        'val_results': val_results,
        'precision_threshold': precision_threshold,
        'business_interpretation': business_interpretation,
        'feature_importance': feature_importance,
        'best_params': trainer.best_params,
        'n_features': X_train.shape[1],
        'model_path': str(model_path) if save_model else None
    }

    logger.info("Production model training pipeline completed")
    logger.info(f"Validation PR-AUC: {val_results['pr_auc']:.4f}")

    return results

if __name__ == "__main__":
    # Train XGBoost production model
    xgb_results = train_production_model('xgboost', n_trials=20)  # Reduced trials for demo

    # Train LightGBM production model
    lgb_results = train_production_model('lightgbm', n_trials=20)

    # Compare results
    print("\n" + "="*50)
    print("PRODUCTION MODEL COMPARISON")
    print("="*50)
    print(f"XGBoost PR-AUC: {xgb_results['val_results']['pr_auc']:.4f}")
    print(f"LightGBM PR-AUC: {lgb_results['val_results']['pr_auc']:.4f}")

    # Choose the better model
    if xgb_results['val_results']['pr_auc'] > lgb_results['val_results']['pr_auc']:
        best_model = 'xgboost'
        best_pr_auc = xgb_results['val_results']['pr_auc']
    else:
        best_model = 'lightgbm'
        best_pr_auc = lgb_results['val_results']['pr_auc']

    print(f"\nBest production model: {best_model} (PR-AUC: {best_pr_auc:.4f})")
