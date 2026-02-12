import time
import requests
import threading
from src.api import app

def run_app():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Start the Flask app in a separate thread
t = threading.Thread(target=run_app)
t.daemon = True
t.start()

# Wait for the app to start
time.sleep(2)

# Test data
data = {'temperature': 85.0, 'vibration': 2.5, 'pressure': 15.0}

# Measure latency
start = time.time()
response = requests.post('http://localhost:5000/predict', json=data)
latency = (time.time() - start) * 1000

print(f'API Response: {response.json()}')
print(f'Latency: {latency:.2f} ms')
print('Latency < 50ms:', latency < 50)
