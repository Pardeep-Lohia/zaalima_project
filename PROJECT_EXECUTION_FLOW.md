# FactoryGuard AI - Project Execution Flow Guide

## 🎯 **Overview**
This guide explains the complete execution flow of the FactoryGuard AI predictive maintenance system. The project follows a modular, end-to-end ML pipeline that transforms raw IoT sensor data into production-ready failure predictions.

## 🚀 **Main Entry Point**
The primary execution starts with `src/complete_pipeline.py`, which orchestrates the entire ML pipeline through the `run_complete_pipeline()` function.

```python
from src.complete_pipeline import run_complete_pipeline

results = run_complete_pipeline(
    regenerate_data=True,      # Generate fresh synthetic data
    model_type='xgboost',      # Production model choice
    tune_hyperparams=True,     # Enable hyperparameter tuning
    n_trials=20,              # Optimization trials
    save_all=True             # Save models, plots, reports
)
```

## 🔄 **Complete Execution Flow**

### **Phase 1: Data Preparation**
**File**: `src/data_preprocessing.py`
**Function**: `load_and_preprocess_data()`

1. **Data Generation/Loading**
   - If `regenerate_data=True`: Generate synthetic IoT data using `generate_synthetic_iot_data()`
   - Creates 10,000 samples with realistic sensor patterns (temperature, vibration, pressure)
   - Introduces failure precursors: gradual degradation over 12-48 hours
   - Failure rate: ~0.56% (highly imbalanced)

2. **Target Creation**
   - Shift failure labels 24 hours back: `failure_24h = failure.shift(-24)`
   - Drop last 24 rows (NaN values after shifting)
   - Result: Predictive target for 24-hour advance warning

3. **Time-Aware Splitting**
   - Chronological train/validation split (70/30)
   - **Critical**: Prevents data leakage by maintaining temporal order
   - Training: First 70% chronologically
   - Validation: Last 30% chronologically

4. **Class Distribution Analysis**
   - Log positive/negative sample counts
   - Report class imbalance statistics

**Output**: `train_df`, `val_df` (pandas DataFrames)

---

### **Phase 2: Feature Engineering**
**File**: `src/feature_engineering.py`
**Function**: `create_feature_pipeline()`

1. **Feature Engineer Class Initialization**
   - Create `FeatureEngineer` instance
   - Define feature creation pipeline

2. **Time-Series Feature Creation**
   - For each sensor (temperature, vibration, pressure):
     - **Rolling Statistics**: Mean, std, variance (1h, 6h, 12h windows)
     - **Trend Analysis**: EMA (12h, 24h), rate of change
     - **Temporal Features**: Lag features (t-1, t-2)
   - **Total Features**: ~45 per sensor × 3 sensors = 42 features

3. **Data Quality Handling**
   - NaN handling: Forward/backward fill for rolling features
   - Drop rows with remaining NaN after fill operations

4. **Scaling**
   - Fit `StandardScaler` on training data only
   - Transform both train and validation sets
   - **Critical**: Prevents data leakage

5. **Pipeline Serialization**
   - Save feature engineering pipeline to `models/feature_pipeline.joblib`

**Output**: `train_features`, `val_features` (scaled DataFrames with 42 features)

---

### **Phase 3: Baseline Model Training**
**File**: `src/train_baseline.py`
**Function**: `train_baseline_models()`

1. **Model Preparation**
   - Extract feature columns (exclude timestamp, failure, failure_24h)
   - Prepare X_train, y_train, X_val, y_val

2. **Logistic Regression Training**
   - `BaselineModelTrainer('logistic_regression')`
   - Class weights: 'balanced'
   - Solver: 'liblinear', max_iter=1000

3. **Random Forest Training**
   - `BaselineModelTrainer('random_forest')`
   - 100 trees, max_depth=10, min_samples_split=10
   - Class weights: 'balanced'

4. **Model Evaluation**
   - **Primary Metric**: PR-AUC (appropriate for imbalanced data)
   - Secondary: Precision, Recall, F1-Score
   - Confusion matrices and PR curves

5. **Feature Importance**
   - LR: Absolute coefficient values
   - RF: Gini importance scores

6. **Model Serialization**
   - Save to `models/baseline_logistic_regression.joblib`
   - Save to `models/baseline_random_forest.joblib`

7. **Plot Generation**
   - PR curves: `reports/baseline_lr_pr_curve.png`
   - Confusion matrices: `reports/baseline_lr_confusion_matrix.png`
   - Feature importance CSVs

**Output**: Baseline results dictionary with metrics and model comparisons

---

### **Phase 4: Production Model Training**
**File**: `src/train_production_model.py`
**Function**: `train_production_model()`

1. **Model Selection**
   - XGBoost or LightGBM based on `model_type` parameter

2. **Hyperparameter Tuning Setup**
   - If `tune_hyperparams=True`: Initialize Optuna study
   - Objective function: Maximize PR-AUC
   - Cross-validation: 3-fold TimeSeriesSplit

3. **Bayesian Optimization**
   - `n_trials` optimization iterations (default: 50)
   - Tune learning_rate, max_depth, n_estimators, subsample, etc.
   - Early stopping to prevent overfitting

4. **Best Model Training**
   - Train final model with best hyperparameters
   - Use class weights for imbalanced data

5. **Model Evaluation**
   - PR-AUC, precision, recall, F1-score
   - Feature importance analysis

6. **Model Serialization**
   - Save to `models/production_xgboost.joblib` or `production_lightgbm.joblib`

7. **Plot Generation**
   - PR curves and confusion matrices
   - Feature importance plots
   - Best parameters CSV

**Output**: Production model results with tuned hyperparameters and performance metrics

---

### **Phase 5: Threshold Optimization**
**File**: `src/threshold_optimization.py`
**Function**: `optimize_thresholds()`

1. **Threshold Analysis**
   - Analyze prediction probabilities vs. true labels
   - Find optimal thresholds for different scenarios

2. **Business Constraints**
   - Fixed recall targets (e.g., 90% recall)
   - Daily alert budget limits
   - Cost-sensitive optimization

3. **Threshold Selection**
   - Balance precision vs. recall
   - Minimize false positives (costly maintenance calls)
   - Maximize true positives (prevent failures)

**Output**: Optimized threshold recommendations

---

### **Phase 6: Enhanced Evaluation**
**File**: `src/enhanced_evaluation.py`
**Functions**: `RareEventEvaluator`, `compare_models_rare_events()`

1. **Rare Event Evaluator Creation**
   - Initialize evaluators for baseline and production models
   - Focus on imbalanced classification metrics

2. **Business KPI Calculation**
   - Alert rates, cost-benefit analysis
   - Maintenance planning impact
   - ROI calculations

3. **Model Comparison**
   - Compare all trained models
   - Rank by PR-AUC and business metrics
   - Identify best performing model

4. **Executive Summary Generation**
   - Business-friendly report
   - Cost savings projections
   - Implementation recommendations

**Output**: Comprehensive evaluation reports and executive summary

---

### **Phase 7: Report Generation**
**Files**: Various evaluation modules

1. **CSV Reports**
   - Model comparison: `reports/model_comparison.csv`
   - Feature importance: `reports/*_feature_importance.csv`
   - Rare event metrics: `reports/*_rare_event_metrics.csv`

2. **Visualization Plots**
   - PR curves: `reports/*_pr_curve.png`
   - Confusion matrices: `reports/*_confusion_matrix.png`
   - Comprehensive evaluation: `reports/*_comprehensive_evaluation.png`

3. **Executive Summary**
   - Business impact analysis: `reports/executive_summary.md`
   - Cost-benefit calculations
   - Implementation roadmap

---

### **Phase 8: Final Results Compilation**
**File**: `src/complete_pipeline.py`

1. **Results Aggregation**
   - Compile all phase outputs into single results dictionary
   - Include data info, model results, evaluations

2. **Pipeline Summary**
   - Print execution summary to console
   - Show best model and performance metrics
   - Provide file paths for detailed reports

**Final Output**: Complete results dictionary with all pipeline artifacts

## 📊 **Data Flow Diagram**

```
Raw Data Generation
        ↓
Time-Aware Split (70/30)
        ↓
Feature Engineering (42 features)
        ↓
Baseline Training (LR + RF)
        ↓
Production Training (XGBoost/LightGBM)
        ↓
Threshold Optimization
        ↓
Enhanced Evaluation
        ↓
Report Generation
        ↓
Executive Summary
```

## 🏃 **Quick Execution Commands**

### **Full Pipeline (Recommended)**
```bash
# Complete end-to-end execution
python -c "from src.complete_pipeline import run_complete_pipeline, print_pipeline_summary; results = run_complete_pipeline(regenerate_data=True, model_type='xgboost', tune_hyperparams=True, n_trials=20, save_all=True); print_pipeline_summary(results)"
```

### **Step-by-Step Execution**
```python
# 1. Data preparation
from src.data_preprocessing import load_and_preprocess_data
train_df, val_df = load_and_preprocess_data()

# 2. Feature engineering
from src.feature_engineering import create_feature_pipeline
train_features, val_features = create_feature_pipeline(train_df, val_df)

# 3. Baseline models
from src.train_baseline import train_baseline_models
baseline_results = train_baseline_models()

# 4. Production model
from src.train_production_model import train_production_model
prod_results = train_production_model('xgboost', n_trials=20)
```

### **Inference (Production Use)**
```python
# Load trained model
from joblib import load
from src.feature_engineering import FeatureEngineer

model = load('models/production_xgboost.joblib')
feature_engineer = FeatureEngineer()
feature_engineer.load_pipeline('models/feature_pipeline.joblib')

# Process new data
processed_data = feature_engineer.transform_features(new_sensor_data, scale=True)
feature_cols = [col for col in processed_data.columns if col not in ['timestamp', 'failure', 'failure_24h']]
predictions = model.predict_proba(processed_data[feature_cols])
failure_probabilities = predictions[:, 1]
```

## ⏱️ **Execution Time Estimates**

| Phase | Time Estimate | Description |
|-------|---------------|-------------|
| Data Preparation | 1-2 minutes | Generate/load and preprocess data |
| Feature Engineering | 2-3 minutes | Create 42 time-series features |
| Baseline Training | 3-5 minutes | Train LR + RF with evaluation |
| Production Training | 15-45 minutes | XGBoost with 50 Optuna trials |
| Threshold Optimization | 1-2 minutes | Find optimal decision boundaries |
| Enhanced Evaluation | 2-3 minutes | Generate business reports |
| Report Generation | 1-2 minutes | Create plots and CSVs |
| **Total** | **25-65 minutes** | Complete pipeline execution |

## 🔍 **Key Decision Points**

1. **Data Regeneration**: `regenerate_data` parameter controls synthetic data creation
2. **Model Choice**: `model_type` selects XGBoost vs LightGBM
3. **Hyperparameter Tuning**: `tune_hyperparams` enables/disables Optuna optimization
4. **Trial Count**: `n_trials` affects tuning thoroughness vs. execution time
5. **Artifact Saving**: `save_all` controls model/plot generation

## 📁 **Output Directory Structure**

```
FactoryGuard AI/
├── models/                          # Serialized models
│   ├── feature_pipeline.joblib      # Feature engineering pipeline
│   ├── baseline_logistic_regression.joblib
│   ├── baseline_random_forest.joblib
│   └── production_xgboost.joblib     # Best production model
├── reports/                         # Evaluation outputs
│   ├── model_comparison.csv         # Performance comparison
│   ├── executive_summary.md         # Business recommendations
│   ├── *_pr_curve.png              # Precision-Recall curves
│   ├── *_confusion_matrix.png      # Confusion matrices
│   ├── *_feature_importance.csv    # Feature rankings
│   └── *_rare_event_metrics.csv    # Business KPIs
└── data/                           # Processed datasets
    ├── synthetic_iot_data.csv      # Raw synthetic data
    ├── train_raw.csv               # Training split
    └── val_raw.csv                 # Validation split
```

## 🎯 **Success Indicators**

- **Data Pipeline**: No NaN values, proper temporal ordering
- **Feature Engineering**: 42 features created without errors
- **Model Training**: PR-AUC > 0.80 for baselines, > 0.85 for production
- **Evaluation**: Comprehensive reports generated
- **Business Value**: Clear cost savings projections in executive summary

---

**The FactoryGuard AI execution flow transforms IoT sensor data into actionable maintenance predictions through a systematic, production-ready ML pipeline.**
