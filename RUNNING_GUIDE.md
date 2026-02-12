# FactoryGuard AI - Running and Demonstration Guide

## Overview

FactoryGuard AI is a predictive maintenance system that uses IoT sensor data to predict equipment failures 24 hours in advance. This guide provides step-by-step instructions for running the complete ML pipeline and demonstrating the system's capabilities.

## Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+
- Git (optional, for cloning)

### Installation
```bash
# Clone or download the project
# cd into the project directory

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
# Run the full pipeline: data generation → feature engineering → model training → evaluation
python test_run.py
```

This will:
- Generate synthetic IoT data (10,000 samples, ~0.5% failure rate)
- Create 135 time-series features from 3 sensors
- Train baseline models (Logistic Regression, Random Forest)
- Train production XGBoost model with hyperparameter tuning
- Generate comprehensive evaluation reports

## Detailed Running Instructions

### Step 1: Environment Setup
```bash
# Ensure you're in the project root directory
cd /path/to/zaalima_project

# Create virtual environment (recommended)
python -m venv factoryguard_env
source factoryguard_env/bin/activate  # On Windows: factoryguard_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Individual Components

#### Data Generation Only
```python
from src.data_preprocessing import generate_synthetic_iot_data
data = generate_synthetic_iot_data()
print(f"Generated {len(data)} samples with {data['failure'].sum()} failures")
```

#### Feature Engineering Only
```python
from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import create_feature_pipeline

train_df, val_df = load_and_preprocess_data()
train_features, val_features = create_feature_pipeline(train_df, val_df)
print(f"Created {train_features.shape[1]} features")
```

#### Train Baseline Models Only
```python
from src.train_baseline import train_baseline_models
results = train_baseline_models()
print(f"Best baseline: {results['best_model']} (PR-AUC: {results['best_pr_auc']:.4f})")
```

#### Train Production Model Only
```python
from src.train_production_model import train_production_model
results = train_production_model('xgboost', n_trials=10)  # Reduce trials for faster execution
print(f"Production model PR-AUC: {results['val_results']['pr_auc']:.4f}")
```

### Step 3: View Results

#### Generated Files
After running the pipeline, check these directories:

```
data/
├── synthetic_iot_data.csv          # Raw sensor data
├── train_raw.csv                   # Training split
└── val_raw.csv                     # Validation split

models/
├── feature_pipeline.joblib         # Feature engineering pipeline
├── baseline_logistic_regression.joblib
├── baseline_random_forest.joblib
└── production_xgboost.joblib       # Tuned production model

reports/
├── executive_summary.md            # Business-friendly summary
├── model_comparison.csv            # Performance metrics
├── *_pr_curve.png                  # Precision-Recall curves
├── *_confusion_matrix.png          # Confusion matrices
└── *_feature_importance.csv        # Feature importance rankings
```

#### Key Metrics to Check
- **PR-AUC**: Primary metric (target: >0.8 for good performance)
- **Failure Recall**: Percentage of failures detected
- **Precision**: Accuracy of alerts (minimizes false alarms)

## Demonstration Scenarios

### Scenario 1: Quick Demo (5 minutes)
```bash
# Fast execution with reduced trials
python -c "from src.complete_pipeline import run_complete_pipeline; results = run_complete_pipeline(regenerate_data=True, model_type='xgboost', tune_hyperparams=True, n_trials=5, save_all=True)"
```

### Scenario 2: Production Simulation
```python
from joblib import load
from src.feature_engineering import FeatureEngineer

# Load trained model
model = load('models/production_xgboost.joblib')
feature_engineer = FeatureEngineer()
feature_engineer.load_pipeline('models/feature_pipeline.joblib')

# Simulate new sensor readings
import pandas as pd
import numpy as np

new_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
    'temperature': np.random.normal(75, 5, 100),
    'vibration': np.random.normal(0.5, 0.1, 100),
    'pressure': np.random.normal(100, 10, 100),
    'failure': 0
})

# Process and predict
processed_data = feature_engineer.transform_features(new_data, scale=True)
features = processed_data.drop(['timestamp', 'failure'], axis=1, errors='ignore')
predictions = model.predict_proba(features)
failure_probabilities = predictions[:, 1]

# Apply threshold (adjust based on business needs)
threshold = 0.3
alerts = (failure_probabilities >= threshold).astype(int)
print(f"Generated {alerts.sum()} maintenance alerts from {len(alerts)} predictions")
```

### Scenario 3: Model Comparison
```python
from src.enhanced_evaluation import compare_models_rare_events

# Load results from reports/model_comparison.csv
import pandas as pd
comparison_df = pd.read_csv('reports/model_comparison.csv')
print(comparison_df)
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors
```
Error: Out of memory during training
Solution: Reduce n_trials parameter
python -c "from src.train_production_model import train_production_model; train_production_model('xgboost', n_trials=5)"
```

#### 2. Poor Model Performance
```
Symptoms: PR-AUC < 0.1, failure recall = 0
Possible causes:
- Data generation issues
- Feature engineering problems
- Class imbalance not handled properly
```

**Debug steps:**
```python
# Check data distribution
from src.data_preprocessing import load_and_preprocess_data
train_df, val_df = load_and_preprocess_data()
print("Training failure rate:", train_df['failure_24h'].mean())
print("Validation failure rate:", val_df['failure_24h'].mean())

# Check feature creation
from src.feature_engineering import create_feature_pipeline
train_features, val_features = create_feature_pipeline(train_df, val_df)
print("Features shape:", train_features.shape)
print("Feature names:", train_features.columns.tolist()[:5])
```

#### 3. Import Errors
```
Error: ModuleNotFoundError
Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

#### 4. Long Training Times
```
Solution: Use LightGBM instead of XGBoost or reduce trials
python -c "from src.train_production_model import train_production_model; train_production_model('lightgbm', n_trials=10)"
```

### Performance Optimization

#### Faster Execution
- Use `n_trials=5-10` instead of 50
- Switch to LightGBM: `model_type='lightgbm'`
- Skip data regeneration: `regenerate_data=False`

#### Memory Usage
- Process data in smaller batches
- Use `joblib` parallel processing (already enabled)
- Monitor with system task manager

## Expected Results

### Current Performance (Based on Testing)
- **PR-AUC**: ~0.0487 (2.4x improvement from 0.02 with recent XGBoost fixes)
- **Failure Detection**: Improved with enhanced features and scale_pos_weight
- **Alert Precision**: Precision-driven threshold optimization (≥60% precision)
- **Business Value**: Cost savings through targeted maintenance alerts

### Realistic Performance Goals
- **PR-AUC**: 0.05-0.10 (achievable with current data and features)
- **24-hour Advance Warning**: Achieved with current implementation
- **Business Value**: Measurable cost savings through reduced false positives

## Business Impact Demonstration

### Cost-Benefit Analysis
```python
# Assuming $50K/hour downtime cost
hourly_downtime_cost = 50000
prevented_failures_per_year = 50  # Estimate based on model performance
annual_savings = prevented_failures_per_year * hourly_downtime_cost

print(f"Estimated annual savings: ${annual_savings:,.0f}")
print(f"ROI: {annual_savings / 100000:.1f}x on $100K implementation cost")
```

### Operational Benefits
- **Predictive Maintenance**: Schedule repairs during off-hours
- **Reduced Downtime**: Minimize unplanned production stops
- **Safety Improvement**: Prevent hazardous failure scenarios
- **Resource Optimization**: Focus maintenance teams effectively

## Advanced Usage

### Custom Data Integration
```python
# Load your own IoT data
# Format: timestamp, temperature, vibration, pressure, failure
your_data = pd.read_csv('your_sensor_data.csv')

# Ensure proper column names and data types
required_columns = ['timestamp', 'temperature', 'vibration', 'pressure', 'failure']
assert all(col in your_data.columns for col in required_columns)

# Save as synthetic data for pipeline
your_data.to_csv('data/synthetic_iot_data.csv', index=False)

# Run pipeline with your data
results = run_complete_pipeline(regenerate_data=False)  # Use existing data
```

### Threshold Optimization
```python
from src.threshold_optimization import optimize_thresholds

# Load validation predictions
# (This requires running the full pipeline first)
# threshold_results = optimize_thresholds(y_val, y_pred_proba, len(val_df))
```

## Next Steps

1. **Performance Investigation**: Debug why current PR-AUC is lower than documented 0.89
2. **Model Improvement**: Implement additional features or algorithms
3. **Production Deployment**: Containerize and deploy to production environment
4. **Monitoring Setup**: Implement model performance monitoring
5. **Integration**: Connect to real IoT data streams

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed README.md for technical specifications
3. Examine the generated reports/ directory for detailed metrics
4. Check logs for error messages and performance details

---

**FactoryGuard AI** - Transforming Industrial Maintenance with AI
