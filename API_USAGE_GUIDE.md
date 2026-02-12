# FactoryGuard AI API Usage Guide

## Overview

FactoryGuard AI provides a REST API for predictive maintenance predictions based on IoT sensor data. This guide explains how to use the API endpoints for integration, testing, and review purposes.

## API Endpoints

### Base URL
```
http://your-server:5000
```

### 1. Health Check Endpoint

**Endpoint:** `GET /health`

**Description:** Check the health status of the API and verify that models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "pipeline_loaded": true
}
```

**Example Request:**
```bash
curl http://localhost:5000/health
```

### 2. Prediction Endpoint

**Endpoint:** `POST /predict`

**Description:** Make a failure prediction based on sensor readings.

**Request Body:**
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "temperature": 75.5,
  "vibration": 0.8,
  "pressure": 105.2
}
```

**Required Fields:**
- `timestamp`: ISO format datetime string
- `temperature`: Float value (°C)
- `vibration`: Float value (mm/s)
- `pressure`: Float value (psi)

**Response:**
```json
{
  "failure_probability": 0.0234,
  "decision": "SAFE",
  "latency_ms": 15.67
}
```

**Decision Logic:**
- `SAFE`: failure_probability < 0.3
- `ALERT`: failure_probability >= 0.3

## Usage Examples

### Python Example
```python
import requests
import json

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# Make prediction
data = {
    "timestamp": "2024-01-01T12:00:00",
    "temperature": 75.5,
    "vibration": 0.8,
    "pressure": 105.2
}

response = requests.post('http://localhost:5000/predict',
                        json=data,
                        headers={'Content-Type': 'application/json'})

if response.status_code == 200:
    result = response.json()
    print(f"Failure Probability: {result['failure_probability']}")
    print(f"Decision: {result['decision']}")
    print(f"Latency: {result['latency_ms']}ms")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### cURL Examples

#### Health Check
```bash
curl -X GET http://localhost:5000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-01T12:00:00",
    "temperature": 75.5,
    "vibration": 0.8,
    "pressure": 105.2
  }'
```

#### Batch Predictions (if supported)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "timestamp": "2024-01-01T12:00:00",
      "temperature": 75.5,
      "vibration": 0.8,
      "pressure": 105.2
    },
    {
      "timestamp": "2024-01-01T13:00:00",
      "temperature": 76.0,
      "vibration": 0.9,
      "pressure": 105.5
    }
  ]'
```

## Testing and Review Scenarios

### 1. Normal Operation Test
```bash
# Test with normal sensor values
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 75.0, "vibration": 0.5, "pressure": 100.0}'
```

Expected: `decision: "SAFE"`, low failure_probability

### 2. High Risk Test
```bash
# Test with abnormal sensor values
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 90.0, "vibration": 2.0, "pressure": 120.0}'
```

Expected: `decision: "ALERT"`, high failure_probability

### 3. Error Handling Tests

#### Missing Fields
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 75.0, "vibration": 0.5}'
```

Expected: 400 error with message about missing fields

#### Invalid Data Types
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01T12:00:00", "temperature": "invalid", "vibration": 0.5, "pressure": 100.0}'
```

Expected: 400 error with validation message

#### Malformed JSON
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 75.0, "vibration": 0.5, "pressure": 100.0'
```

Expected: 400 error

### 4. Performance Testing

#### Latency Test
```bash
# Run multiple requests to check average latency
for i in {1..10}; do
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 75.0, "vibration": 0.5, "pressure": 100.0}' \
    -w "%{time_total}\n" -o /dev/null
done
```

Expected: Average latency < 50ms

#### Load Test
```bash
# Simple load test with parallel requests
seq 1 100 | xargs -n1 -P10 curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 75.0, "vibration": 0.5, "pressure": 100.0}' \
  -w "%{http_code}\n" -o /dev/null
```

### 5. Integration Testing

#### With Real IoT Data
```python
import pandas as pd
import requests

# Load test data
test_data = pd.read_csv('data/synthetic_iot_data.csv')

# Sample a few rows for testing
sample_data = test_data.head(5)

for _, row in sample_data.iterrows():
    payload = {
        "timestamp": row['timestamp'],
        "temperature": row['temperature'],
        "vibration": row['vibration'],
        "pressure": row['pressure']
    }

    response = requests.post('http://localhost:5000/predict', json=payload)
    print(f"Input: {payload}")
    print(f"Response: {response.json()}")
    print("---")
```

## Review Checklist

### Functionality Review
- [ ] Health endpoint returns correct status
- [ ] Prediction endpoint accepts valid input
- [ ] Prediction returns expected fields
- [ ] Decision logic works correctly (threshold: 0.3)
- [ ] Error handling for invalid inputs
- [ ] Proper HTTP status codes returned

### Performance Review
- [ ] Latency < 50ms for single predictions
- [ ] API handles concurrent requests
- [ ] Memory usage remains stable
- [ ] No memory leaks during extended testing

### Security Review
- [ ] Input validation prevents injection attacks
- [ ] Error messages don't leak sensitive information
- [ ] Rate limiting considerations (if implemented)
- [ ] HTTPS support (for production)

### Documentation Review
- [ ] All endpoints documented
- [ ] Request/response examples provided
- [ ] Error codes explained
- [ ] Authentication requirements noted (if any)

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if the API server is running
   - Verify the correct port (default: 5000)
   - Check firewall settings

2. **Model Loading Errors**
   - Ensure model files exist in `models/` directory
   - Check file permissions
   - Verify model compatibility

3. **High Latency**
   - Check system resources (CPU, memory)
   - Review feature engineering pipeline
   - Consider model optimization

4. **Invalid Predictions**
   - Verify input data format
   - Check feature engineering consistency
   - Review model training data distribution

## Production Considerations

### Monitoring
- Implement health check monitoring
- Set up latency tracking
- Monitor prediction accuracy over time

### Scaling
- Consider using Gunicorn for production deployment
- Implement request queuing for high load
- Use load balancer for multiple instances

### Security
- Implement authentication/authorization
- Use HTTPS in production
- Validate and sanitize all inputs
- Implement rate limiting

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs for error details
3. Test with the provided examples
4. Contact the development team for advanced issues

---

**FactoryGuard AI** - Predictive Maintenance API Guide
