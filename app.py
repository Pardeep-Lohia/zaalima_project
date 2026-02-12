import time
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from pathlib import Path
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and pipeline
MODEL = None
FEATURE_PIPELINE = None
FEATURE_COLUMNS = None
THRESHOLD = 0.3

def load_model_and_pipeline():
    """
    Load the production model and feature engineering pipeline globally.
    """
    global MODEL, FEATURE_PIPELINE, FEATURE_COLUMNS

    try:
        # Load production model
        model_path = Path('models/production_xgboost.joblib')
        model_data = joblib.load(model_path)
        MODEL = model_data['model']
        FEATURE_COLUMNS = model_data['feature_columns']
        logger.info(f"Loaded production model from {model_path}")

        # Load feature engineering pipeline
        from src.feature_engineering import FeatureEngineer
        pipeline_path = Path('models/feature_pipeline.joblib')
        pipeline_data = joblib.load(pipeline_path)

        # Recreate FeatureEngineer object from saved data
        FEATURE_PIPELINE = FeatureEngineer(sensor_columns=pipeline_data['sensor_columns'])
        FEATURE_PIPELINE.scaler = pipeline_data['scaler']
        FEATURE_PIPELINE.is_fitted = pipeline_data['is_fitted']

        logger.info(f"Loaded feature engineering pipeline from {pipeline_path}")

    except Exception as e:
        logger.error(f"Failed to load model or pipeline: {e}")
        raise

# Load model and pipeline on startup
load_model_and_pipeline()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'pipeline_loaded': FEATURE_PIPELINE is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for IoT predictive maintenance.
    """
    start_time = time.time()

    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate required fields
        required_fields = ['timestamp', 'temperature', 'vibration', 'pressure']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create DataFrame from input
        input_df = pd.DataFrame([{
            'timestamp': pd.to_datetime(data['timestamp']),
            'temperature': float(data['temperature']),
            'vibration': float(data['vibration']),
            'pressure': float(data['pressure'])
        }])

        # Transform input using feature engineering pipeline
        transformed_df = FEATURE_PIPELINE.transform_features(input_df, scale=True)

        # Prepare features for prediction - ensure same order as training
        feature_cols = [col for col in transformed_df.columns
                       if col not in ['timestamp', 'failure', 'failure_24h']]
        X_pred = transformed_df[feature_cols]

        # Ensure feature columns match training order
        if FEATURE_COLUMNS:
            # Reorder columns to match training
            X_pred = X_pred.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # Make prediction
        failure_probability = float(MODEL.predict_proba(X_pred)[0][1])

        # Make decision based on threshold
        decision = 'ALERT' if failure_probability >= THRESHOLD else 'SAFE'

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Return response
        response = {
            'failure_probability': round(failure_probability, 4),
            'decision': decision,
            'latency_ms': round(latency_ms, 2)
        }

        logger.info(f"Prediction completed in {latency_ms:.2f}ms - Decision: {decision}")
        return jsonify(response)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
