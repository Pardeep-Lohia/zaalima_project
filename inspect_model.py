import joblib

# Load the model
model = joblib.load('models/production_xgboost.joblib')

print('Model keys:', list(model.keys()))
if 'feature_columns' in model:
    print('Feature columns length:', len(model['feature_columns']))
    print('First 5 feature columns:', model['feature_columns'][:5])
    print('Last 5 feature columns:', model['feature_columns'][-5:])
else:
    print('No feature_columns in model')

# Load pipeline
pipeline = joblib.load('models/feature_pipeline.joblib')
print('Pipeline keys:', list(pipeline.keys()))
