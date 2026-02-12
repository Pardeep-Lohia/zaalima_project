import time
import json
from flask import Flask, request, jsonify
from joblib import load
from pathlib import Path
import pandas as pd
import numpy as np
from .utils import get_project_root

app = Flask(__name__)

# Load model and pipeline at startup
MODEL_PATH = Path(get_project_root()) / 'models' / 'production_xgboost.joblib'
PIPELINE_PATH = Path(get_project_root()) / 'models' / 'feature_pipeline.joblib'

try:
    model = load(MODEL_PATH)
    feature_engineer = load(PIPELINE_PATH)
    print("Model and pipeline loaded successfully")
except Exception as e:
    print(f"Error loading model or pipeline: {e}")
    model = None
    feature_engineer = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict failure probability for sensor readings.

    Expected JSON payload:
    {
        "temperature": float,
        "vibration": float,
        "pressure": float
    }
    """
    start_time = time.time()

    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate required fields
        required_fields = ['temperature', 'vibration', 'pressure']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create DataFrame with current timestamp
        current_time = pd.Timestamp.now()
        sensor_data = pd.DataFrame({
            'timestamp': [current_time],
            'temperature': [data['temperature']],
            'vibration': [data['vibration']],
            'pressure': [data['pressure']],
            'failure': [0]  # Not used for prediction
        })

        # Process through feature engineering pipeline
        processed_data = feature_engineer.transform_features(sensor_data, scale=True)

        # Get feature columns (exclude timestamp, failure, failure_24h)
        feature_cols = [col for col in processed_data.columns
                       if col not in ['timestamp', 'failure', 'failure_24h']]
        features = processed_data[feature_cols]

        # Make prediction
        probabilities = model.predict_proba(features)
        failure_probability = float(probabilities[0][1])

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        response = {
            'failure_probability': failure_probability,
            'prediction': 'high_risk' if failure_probability > 0.5 else 'low_risk',
            'latency_ms': round(latency_ms, 2),
            'timestamp': current_time.isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'pipeline_loaded': feature_engineer is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
