import pandas as pd
from src.train_production_model import train_production_model
from src.threshold_optimization import ThresholdOptimizer
from src.enhanced_evaluation import compare_models_rare_events

# Train improved XGBoost model
print('Training improved XGBoost model...')
xgb_results = train_production_model('xgboost', n_trials=10, save_model=False, save_plots=False)

# Get validation predictions for threshold optimization
from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import create_feature_pipeline
from pathlib import Path
import joblib

train_df, val_df = load_and_preprocess_data()
train_features, val_features = create_feature_pipeline(train_df, val_df)

feature_cols = [col for col in train_features.columns if col not in ['timestamp', 'failure', 'failure_24h']]
X_val = val_features[feature_cols]
y_val = val_features['failure_24h']

# Load trained model
model_path = Path('models/production_xgboost.joblib')
if model_path.exists():
    model_data = joblib.load(model_path)
    model = model_data['model']
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Precision-driven threshold optimization
    optimizer = ThresholdOptimizer(y_val, y_pred_proba)
    precision_threshold = optimizer.find_threshold_for_precision(min_precision=0.6)

    print(f'Precision-driven threshold: {precision_threshold["threshold"]:.4f}')
    print(f'Precision: {precision_threshold["precision"]:.4f}')
    print(f'Recall: {precision_threshold["recall"]:.4f}')
    print(f'Alerts per day: {precision_threshold["alert_rate"] * 24:.1f}')
    print(f'False positives per 1000 predictions: {(precision_threshold["false_positives"] / len(y_val)) * 1000:.2f}')

    # Add to results
    xgb_results['precision_threshold'] = precision_threshold

    # Model comparison
    model_comparison = compare_models_rare_events({'xgboost': xgb_results}, len(y_val))
    print('\nModel Comparison:')
    print(model_comparison.to_markdown(index=False))
else:
    print('Model not found, training...')
    xgb_results = train_production_model('xgboost', n_trials=10, save_model=True, save_plots=False)
