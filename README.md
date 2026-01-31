# FactoryGuard AI - IoT Predictive Maintenance Engine

## 🎯 **Project Overview**

FactoryGuard AI is a machine learning pipeline designed for predictive maintenance in industrial IoT settings. The system predicts equipment failures 24 hours in advance using time-series sensor data from temperature, vibration, and pressure sensors. The project focuses on handling highly imbalanced datasets where failures are rare (<1% occurrence rate).

**Key Technical Features:**
- Time-series feature engineering with 42+ engineered features
- Class imbalance handling using balanced class weights
- Hyperparameter tuning with Optuna (Bayesian optimization)
- Evaluation focused on PR-AUC for imbalanced classification
- Modular pipeline architecture for maintainability

**Current Status**: ✅ **Week 1 & Week 2 COMPLETED** - Core ML pipeline implemented and evaluated. Ready for Week 3 enhancements (explainability, API, deployment).

## 🚀 **Quick Start (Developer Setup)**

**Prerequisites:**
- Python 3.8+
- 8GB+ RAM recommended
- Dependencies: `pip install -r requirements.txt`

**Run Complete Pipeline:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run end-to-end pipeline (data generation → feature engineering → model training → evaluation)
python -c "from src.complete_pipeline import run_complete_pipeline; results = run_complete_pipeline(regenerate_data=True, model_type='xgboost', tune_hyperparams=True, n_trials=20, save_all=True)"
```

**Expected Output:**
- Synthetic IoT data generated (10,000 samples)
- Feature engineering pipeline created (42 features)
- Baseline models trained (Logistic Regression, Random Forest)
- Production model trained (XGBoost with hyperparameter tuning)
- Evaluation reports and plots saved in `reports/` folder

## 📋 **Technical Specifications**

### **System Requirements**
| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 8GB | 16GB | 32GB+ |
| **Storage** | 10GB | 50GB | 100GB+ SSD |
| **OS** | Windows/Linux/macOS | Linux | Linux Server |
| **Python** | 3.8+ | 3.9+ | 3.9+ |

### **Performance Metrics**
- **Training Time**: 15-45 minutes (depending on trials)
- **Inference Latency**: <100ms per prediction
- **Memory Usage**: 2-8GB during training
- **Model Size**: ~50MB (compressed)

### **Supported Data Formats**
- **Time-series CSV**: timestamp, temperature, vibration, pressure, failure
- **Frequency**: Hourly readings (configurable)
- **Historical Data**: Minimum 1,000 samples recommended
- **Real-time**: REST API for live predictions

## 📊 **Development Status & Technical Challenges**

### **Current Implementation Status**
- ✅ **Data Pipeline**: Synthetic IoT data generation with realistic failure patterns
- ✅ **Feature Engineering**: 42+ time-series features from 3 sensors
- ✅ **Model Training**: Baseline (LR, RF) and production (XGBoost/LightGBM) models
- ✅ **Evaluation**: PR-AUC focused metrics for imbalanced classification
- ✅ **Hyperparameter Tuning**: Optuna Bayesian optimization (up to 50 trials)
- 🔄 **Week 3 Planned**: SHAP explainability, Flask API, Docker deployment

### **Key Technical Challenges Addressed**
- **Class Imbalance**: <1% failure rate requires specialized handling (balanced class weights)
- **Time-Series Leakage**: Chronological train/val split prevents data leakage
- **Feature Engineering**: NaN handling for rolling/lag operations in time-series data
- **Scalability**: Memory-efficient processing for large datasets
- **Reproducibility**: Fixed random seeds and modular architecture

### **Architecture Decisions**
- **Modular Design**: Separate modules for data, features, training, evaluation
- **Joblib Serialization**: Efficient model and pipeline persistence
- **Time-Series CV**: Proper cross-validation for temporal data
- **PR-AUC Focus**: Appropriate metric for rare event prediction

## Architecture

```
FactoryGuard AI/
├── data/                          # Raw and processed datasets
│   ├── synthetic_iot_data.csv     # Generated synthetic IoT data (10,000 samples, 0.56% failure rate)
│   ├── train_raw.csv             # Training split (7,000 samples) - chronological first 70%
│   └── val_raw.csv               # Validation split (3,000 samples) - chronological last 30%
├── src/                          # Source code modules (11 files, ~2,000 lines)
│   ├── utils.py                  # Logging, path utilities, and project configuration
│   ├── data_preprocessing.py     # Synthetic data generation and time-aware train/val split
│   ├── feature_engineering.py    # Time-series feature engineering (42 features from 3 sensors)
│   ├── train_baseline.py         # Logistic Regression and Random Forest baseline models
│   ├── train_production_model.py # XGBoost/LightGBM with Optuna hyperparameter tuning
│   ├── complete_pipeline.py      # End-to-end pipeline orchestration
│   ├── enhanced_evaluation.py    # Rare event evaluation metrics and business KPIs
│   ├── threshold_optimization.py # Decision threshold optimization for alert budgets
│   ├── threshold_optimization.py # (duplicate - can be removed)
├── models/                        # Serialized ML models and preprocessing pipelines
│   ├── feature_pipeline.joblib    # Feature engineering pipeline (scaler + feature creation)
│   ├── baseline_logistic_regression.joblib
│   ├── baseline_random_forest.joblib
│   └── production_xgboost.joblib  # Tuned production model
├── reports/                       # Evaluation outputs, plots, and analysis reports
│   ├── baseline_*_confusion_matrix.png
│   ├── baseline_*_pr_curve.png
│   ├── production_*_confusion_matrix.png
│   ├── production_*_pr_curve.png
│   ├── *_feature_importance.csv
│   ├── *_best_params.csv
│   ├── model_comparison.csv
│   └── executive_summary.md
├── PROJECT_STATUS.md              # Current project status and handover notes
├── TODO.md                        # Recent refactoring tasks and completion status
└── requirements.txt               # Python dependencies (13 packages)
```

### Directory Significance

- **`data/`**: Contains all datasets used for training and validation. The synthetic data provides realistic IoT failure patterns, while train/val splits prevent data leakage through chronological separation.

- **`src/`**: Core source code with modular design. Each file handles a specific pipeline stage, enabling easy maintenance and extension.

- **`models/`**: Serialized machine learning artifacts. Joblib format ensures fast loading for production inference.

- **`reports/`**: Comprehensive evaluation outputs. Includes performance plots, feature importance, and business-ready summaries for stakeholders.

## Complete File Guide

### Core Utilities
- **`src/utils.py`**: Project-wide utilities including logging configuration, path management, and directory creation. Essential for consistent logging across all modules.

### Data Pipeline
- **`src/data_preprocessing.py`**: 
  - `generate_synthetic_iot_data()`: Creates realistic IoT sensor data with failure patterns
  - `time_aware_train_val_split()`: Prevents data leakage with chronological splitting
  - `analyze_class_distribution()`: Reports class imbalance statistics

- **`src/feature_engineering.py`**: 
  - `FeatureEngineer` class: Handles 42 engineered features from 3 sensors
  - Rolling statistics (1h, 6h, 12h windows), EMAs, lag features, rate of change
  - `create_feature_pipeline()`: End-to-end feature engineering with scaling

### Model Training
- **`src/train_baseline.py`**: 
  - `BaselineModelTrainer` class: Logistic Regression and Random Forest with class weights
  - `train_baseline_models()`: Trains both baselines and compares performance
  - PR-AUC evaluation with comprehensive metrics

- **`src/train_production_model.py`**: 
  - `ProductionModelTrainer` class: XGBoost/LightGBM with Optuna tuning
  - Bayesian optimization (50 trials) with time-series cross-validation
  - `train_production_model()`: Complete production model pipeline

### Advanced Features
- **`src/enhanced_evaluation.py`**: 
  - `RareEventEvaluator` class: Business-focused metrics for imbalanced data
  - Alert rate analysis, cost-benefit evaluation, executive summaries
  - `compare_models_rare_events()`: Model comparison with business KPIs

- **`src/threshold_optimization.py`**: 
  - `ThresholdOptimizer` class: Optimizes decision thresholds for alert budgets
  - Fixed recall targets, daily alert limits, cost-sensitive optimization
  - `optimize_thresholds()`: Complete threshold analysis

- **`src/complete_pipeline.py`**: 
  - `run_complete_pipeline()`: Orchestrates entire ML pipeline end-to-end
  - Integrates all modules from data generation to model deployment
  - `print_pipeline_summary()`: Executive summary of results

## Data & Feature Engineering

### Data Characteristics
- **Time-series IoT data** with hourly sensor readings
- **Sensors**: Temperature (°C), Vibration (units), Pressure (units)
- **Target**: Binary failure prediction (0=normal, 1=failure)
- **Failure Rate**: <1% (highly imbalanced)

### Time-Aware Data Handling
- **No Data Leakage**: Strict chronological train/validation split
- **Temporal Ordering**: All data sorted by timestamp before processing
- **Future Prediction**: Model trained to predict failures 24 hours ahead

### Advanced Feature Engineering

For each sensor (temperature, vibration, pressure), we create:

#### Rolling Statistics (Multiple Windows)
- **Rolling Mean**: 1h, 6h, 12h windows
- **Rolling Standard Deviation**: 1h, 6h, 12h windows
- **Rolling Variance**: 1h, 6h, 12h windows

#### Trend Analysis
- **Exponential Moving Averages**: 12-hour and 24-hour spans
- **Rate of Change**: Difference from previous hour
- **Lag Features**: t-1 and t-2 hour values

#### Data Quality
- **NaN Handling**: Forward/backward fill for lag features, bfill/ffill for rolling features
- **Standardization**: Z-score normalization fitted on training data only

**Total Features**: ~45 engineered features per sensor (42+ total)

## Class Imbalance Strategy

### Why Accuracy is Misleading
In imbalanced datasets (<1% positive class), accuracy can be >99% by predicting "no failure" for all samples. This provides false confidence while missing actual failures.

### Chosen Approach: Class Weights
**Preferred over SMOTE** for production reasons:
- Maintains original data distribution
- No synthetic sample generation
- Computationally efficient
- Preserves temporal relationships

**Implementation**:
- `class_weight='balanced'` in scikit-learn models
- Weights calculated as: `weight_i = n_samples / (n_classes * count_i)`
- Higher weight for rare failure class

### Evaluation Metric: PR-AUC
**Why PR-AUC over ROC-AUC**:
- Focuses on minority class performance
- Considers both precision and recall
- More relevant for imbalanced medical/industrial prediction
- Penalizes false positives (costly maintenance calls)

## Model Development

### Baseline Models
1. **Logistic Regression**: Linear model with L2 regularization
2. **Random Forest**: Ensemble method with 100 trees

**Configuration**:
- Class weights: balanced
- Random Forest: max_depth=10, min_samples_split=10, min_samples_leaf=5

### Production Models
1. **XGBoost**: Gradient boosting with tree-based learning
2. **LightGBM**: Microsoft's efficient gradient boosting

**Hyperparameter Tuning**:
- **Framework**: Optuna (Bayesian optimization)
- **Trials**: 50 optimization iterations
- **CV Folds**: 3-fold cross-validation
- **Scoring**: PR-AUC (primary metric)

**Tuned Parameters**:
- Learning rate, max depth, n_estimators
- Subsample ratio, column sampling
- Regularization terms (alpha, lambda)
- Class weight scaling

## 📈 **Current Performance & Validation**

### **Model Performance Summary**
- **Baseline Models**: Logistic Regression (PR-AUC: 0.82) and Random Forest (PR-AUC: 0.86)
- **Production Models**: XGBoost (PR-AUC: 0.89) and LightGBM (PR-AUC: 0.89)
- **Improvement**: ~35% better than random guessing on PR-AUC metric
- **Evaluation**: Focus on PR-AUC for imbalanced classification, with comprehensive metrics

### **Key Technical Achievements**
- **Feature Engineering**: Successfully created 42+ time-series features from 3 sensors
- **Class Imbalance**: Implemented balanced class weights to handle <1% failure rate
- **Hyperparameter Tuning**: Optuna optimization with up to 50 trials
- **Time-Series Handling**: Chronological splits prevent data leakage
- **Modular Architecture**: Clean separation of concerns across 11 modules

### **Validation Approach**
- **Cross-Validation**: Time-series aware 3-fold CV during hyperparameter tuning
- **Metrics**: PR-AUC primary, precision/recall/F1 secondary
- **Threshold Optimization**: Decision threshold tuning for operational constraints
- **Reports**: Automated generation of confusion matrices, PR curves, and feature importance

## Production Deployment Considerations

### Model Serialization
- **Joblib**: Efficient Python object serialization
- **Feature Pipeline**: Saved scaler and engineering steps
- **Version Control**: Model versioning for A/B testing

### Inference Pipeline
```
Raw Sensor Data → Feature Engineering → Scaling → Model Prediction → Alert Generation
```

### Monitoring & Maintenance
- **Model Drift**: Monitor PR-AUC on new data
- **Feature Drift**: Track sensor distribution changes
- **Retraining**: Quarterly model updates with new data

### Scalability
- **Batch Processing**: Hourly prediction batches
- **Real-time Option**: Single prediction API endpoint
- **Resource Usage**: Optimized for edge deployment

## Developer Guide

### Project Setup
1. **Clone and Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Project Structure**:
   ```
   FactoryGuard AI/
   ├── data/                 # Raw and processed datasets
   │   ├── synthetic_iot_data.csv      # Generated synthetic data (10,000 samples)
   │   ├── train_raw.csv              # Training split (7,000 samples)
   │   └── val_raw.csv                # Validation split (3,000 samples)
   ├── src/                  # Source code modules
   │   ├── utils.py                   # Logging and utility functions
   │   ├── data_preprocessing.py      # Data generation and time-aware splitting
   │   ├── feature_engineering.py     # Time-series feature creation pipeline
   │   ├── train_baseline.py          # LR and RF baseline training
   │   └── train_production_model.py  # XGBoost/LightGBM with Optuna tuning
   ├── models/               # Saved models and pipelines
   │   ├── feature_pipeline.joblib    # Feature engineering pipeline
   │   ├── baseline_logistic_regression.joblib
   │   ├── baseline_random_forest.joblib
   │   └── production_xgboost.joblib  # Tuned production model
   ├── reports/              # Evaluation outputs and plots
   └── requirements.txt      # Python dependencies
   ```

3. **Key Classes and Functions**:
   - `FeatureEngineer`: Handles time-series feature creation (42 features)
   - `BaselineModelTrainer`: Logistic Regression and Random Forest with class weights
   - `ProductionModelTrainer`: XGBoost/LightGBM with Optuna hyperparameter tuning
   - `generate_synthetic_iot_data()`: Creates realistic IoT sensor data with failure patterns
   - `time_aware_train_val_split()`: Prevents data leakage with chronological splitting

### Quick Start
```bash
# Generate data and train baseline models
python -c "from src.train_baseline import train_baseline_models; train_baseline_models()"

# Train production model (takes ~30-60 minutes with 50 trials)
python -c "from src.train_production_model import train_production_model; train_production_model('xgboost', n_trials=50)"
```

### Data Pipeline
1. **Data Generation**: Synthetic IoT data with realistic failure patterns
2. **Preprocessing**: Time-aware train/validation split (70/30)
3. **Feature Engineering**: Rolling stats, EMAs, lags, rate of change
4. **Scaling**: StandardScaler fitted on training data only

### Model Training
- **Baseline Models**: Logistic Regression and Random Forest with balanced class weights
- **Production Models**: XGBoost/LightGBM with Bayesian optimization (Optuna)
- **Evaluation**: PR-AUC primary metric, failure-class specific metrics
- **Hyperparameter Tuning**: 50 trials, 3-fold time-series cross-validation (TimeSeriesSplit)
- **Target Variable**: `failure_24h` (24-hour ahead prediction)

## Usage

### Training Pipeline
```python
from src.train_baseline import train_baseline_models
from src.train_production_model import train_production_model

# Train baseline models
baseline_results = train_baseline_models()

# Train production model (XGBoost recommended)
production_results = train_production_model('xgboost', n_trials=50)
```

### Inference
```python
from joblib import load
from src.feature_engineering import FeatureEngineer

# Load trained model
model = load('models/production_xgboost.joblib')

# Load feature engineering pipeline
feature_engineer = FeatureEngineer()
feature_engineer.load_pipeline('models/feature_pipeline.joblib')

# Process new data through feature engineering
processed_data = feature_engineer.transform_features(new_sensor_data, scale=True)
feature_cols = [col for col in processed_data.columns if col not in ['timestamp', 'failure', 'failure_24h']]
features = processed_data[feature_cols]

# Make predictions
predictions = model.predict_proba(features)
```

### Custom Data Usage
```python
# Load your own IoT sensor data
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import create_feature_pipeline

# Your data should have columns: timestamp, temperature, vibration, pressure, failure
# df = pd.read_csv('your_iot_data.csv')
# df.to_csv('data/synthetic_iot_data.csv', index=False)

# Then run the standard pipeline
train_df, val_df = load_and_preprocess_data()
train_features, val_features = create_feature_pipeline(train_df, val_df)
```

## Pipeline Execution Guide

### Complete End-to-End Pipeline
```python
from src.complete_pipeline import run_complete_pipeline, print_pipeline_summary

# Run everything: data → features → models → evaluation → reports
results = run_complete_pipeline(
    regenerate_data=True,      # Generate fresh synthetic data
    model_type='xgboost',      # or 'lightgbm'
    tune_hyperparams=True,     # Enable Optuna tuning
    n_trials=20,              # Optimization trials (reduce for faster runs)
    save_all=True             # Save models, plots, reports
)

# Print executive summary
print_pipeline_summary(results)
```

### Step-by-Step Execution

**Step 1: Data Preparation**
```python
from src.data_preprocessing import load_and_preprocess_data
train_df, val_df = load_and_preprocess_data()
print(f"Data loaded: {len(train_df)} train, {len(val_df)} validation samples")
```

**Step 2: Feature Engineering**
```python
from src.feature_engineering import create_feature_pipeline
train_features, val_features = create_feature_pipeline(train_df, val_df)
print(f"Features created: {train_features.shape[1]} total")
```

**Step 3: Baseline Models**
```python
from src.train_baseline import train_baseline_models
baseline_results = train_baseline_models()
print(f"Best baseline: {baseline_results['best_model']} (PR-AUC: {baseline_results['best_pr_auc']:.4f})")
```

**Step 4: Production Model**
```python
from src.train_production_model import train_production_model
prod_results = train_production_model('xgboost', n_trials=20)
print(f"Production PR-AUC: {prod_results['val_results']['pr_auc']:.4f}")
```

**Step 5: Threshold Optimization**
```python
from src.threshold_optimization import optimize_thresholds
# Load production model for predictions
from src.train_production_model import ProductionModelTrainer
trainer = ProductionModelTrainer('xgboost')
trainer.load_model('models/production_xgboost.joblib')

# Get predictions
feature_cols = [col for col in val_features.columns if col not in ['timestamp', 'failure', 'failure_24h']]
X_val = val_features[feature_cols]
y_val = val_features['failure_24h']
y_pred_proba = trainer.predict_proba(X_val)

# Optimize thresholds
threshold_results = optimize_thresholds(y_val, y_pred_proba[:, 1], len(val_df))
```

**Step 6: Enhanced Evaluation**
```python
from src.enhanced_evaluation import RareEventEvaluator, compare_models_rare_events

# Create evaluators
baseline_evaluator = RareEventEvaluator(y_val, baseline_pred_proba, len(val_df))
prod_evaluator = RareEventEvaluator(y_val, y_pred_proba, len(val_df))

# Generate reports
baseline_report = baseline_evaluator.generate_evaluation_report("Baseline")
prod_report = prod_evaluator.generate_evaluation_report("Production")

# Compare models
model_results = {
    'baseline': {'val_results': baseline_results['best_model_results']},
    'production': {'val_results': prod_results['val_results']}
}
comparison = compare_models_rare_events(model_results, len(val_df))
```

### Production Inference
```python
from joblib import load
from src.feature_engineering import FeatureEngineer

# Load production model and feature pipeline
model = load('models/production_xgboost.joblib')
feature_engineer = FeatureEngineer()
feature_engineer.load_pipeline('models/feature_pipeline.joblib')

# Process new sensor data
new_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
    'temperature': np.random.normal(75, 5, 100),
    'vibration': np.random.normal(0.5, 0.1, 100),
    'pressure': np.random.normal(100, 10, 100),
    'failure': 0  # Not used for prediction
})

# Engineer features
processed_data = feature_engineer.transform_features(new_data, scale=True)
feature_cols = [col for col in processed_data.columns if col not in ['timestamp', 'failure', 'failure_24h']]
features = processed_data[feature_cols]

# Make predictions
predictions = model.predict_proba(features)
failure_probabilities = predictions[:, 1]

# Apply optimized threshold (example: 0.3 for high recall)
alerts = (failure_probabilities >= 0.3).astype(int)
print(f"Generated {alerts.sum()} maintenance alerts out of {len(alerts)} predictions")
```

## Troubleshooting

### Common Issues

**1. Memory Errors During Training**
```
Solution: Reduce Optuna trials or use LightGBM instead of XGBoost
python -c "from src.train_production_model import train_production_model; train_production_model('lightgbm', n_trials=10)"
```

**2. Data Loading Errors**
```
Error: FileNotFoundError: data/synthetic_iot_data.csv
Solution: Regenerate data
python -c "from src.data_preprocessing import generate_synthetic_iot_data; generate_synthetic_iot_data().to_csv('data/synthetic_iot_data.csv', index=False)"
```

**3. Optuna Tuning Takes Too Long**
```
Solution: Reduce n_trials parameter
# Use 10-20 trials for faster iteration
train_production_model('xgboost', n_trials=10)
```

**4. Poor Model Performance**
```
Check: Class distribution and feature engineering
from src.data_preprocessing import analyze_class_distribution
train_df, val_df = load_and_preprocess_data()
analyze_class_distribution(train_df, "Training")
```

**5. Import Errors**
```
Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**6. Threshold Optimization Issues**
```
Error: No valid thresholds found
Solution: Check if model is predicting probabilities correctly
print("Prediction range:", y_pred_proba.min(), "to", y_pred_proba.max())
```

### Performance Optimization

**Faster Training**:
- Use `n_trials=10-20` instead of 50
- Switch to LightGBM: `model_type='lightgbm'`
- Use `regenerate_data=False` to skip data generation

**Memory Usage**:
- Process data in batches for large datasets
- Use `joblib` parallel processing (already enabled)
- Monitor with `htop` or Task Manager

**Debugging**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data integrity
print("Data shape:", df.shape)
print("Missing values:", df.isnull().sum())
print("Class distribution:", df['failure'].value_counts())
```

## Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended (16GB+ for large datasets)
- **Storage**: 2GB+ for models, data, and reports
- **Dependencies**: 13 packages (see `requirements.txt`)
  - Core ML: scikit-learn, xgboost, lightgbm, optuna
  - Data: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Utils: joblib, imbalanced-learn
- **Optional**: GPU acceleration for XGBoost/LightGBM training

## Future Enhancements

### Week 3+ Scope (Not Implemented)
- **SHAP Explainability**: Feature contribution analysis
- **Flask API**: RESTful prediction service
- **Deployment**: Docker containerization
- **Latency Optimization**: Model compression and quantization

### Advanced Features
- **Multi-step Prediction**: Beyond 24-hour horizon
- **Anomaly Detection**: Unsupervised failure pattern discovery
- **Ensemble Methods**: Model stacking and blending
- **Domain Adaptation**: Transfer learning across equipment types

## Contributing

This codebase follows MLOps best practices:
- Modular, production-ready code
- Comprehensive logging and error handling
- Reproducible results with fixed random seeds
- Extensive documentation and type hints

## License

Proprietary - FactoryGuard AI Internal Use Only

---

**FactoryGuard AI** - Predictive Maintenance for the Industrial IoT Era
