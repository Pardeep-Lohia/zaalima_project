# FactoryGuard AI - VPS Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the FactoryGuard AI Flask API to a blank Ubuntu 20.04 VPS. The deployment will place the application in `/var/www/html/factoryguard` and configure it for production use with Gunicorn and Nginx.

## Prerequisites

- Blank Ubuntu 20.04 VPS with root access
- SSH access to the VPS
- Project files copied to your local machine
- Basic knowledge of Linux command line

## Step 1: Initial VPS Setup

### 1.1 Update System and Install Basic Dependencies

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv nginx curl wget git ufw
```

### 1.2 Configure Firewall

```bash
# Enable UFW firewall
sudo ufw enable

# Allow SSH (important - do this first!)
sudo ufw allow ssh

# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Check status
sudo ufw status
```

### 1.3 Create Application Directory

```bash
# Create application directory
sudo mkdir -p /var/www/html/factoryguard

# Set proper ownership (we'll create the user later)
sudo chown -R www-data:www-data /var/www/html/factoryguard
```

## Step 2: Transfer Project Files

### Option A: Using SCP (from your local machine)

```bash
# From your local machine (Windows Command Prompt or PowerShell)
# Replace 'user' with your VPS username and 'your-vps-ip' with actual IP
scp -r C:\Users\Pardeep\Desktop\zaalima_project user@your-vps-ip:/tmp/
```

### Option B: Using Git (if repository is available)

```bash
# On VPS
cd /tmp
git clone https://github.com/your-repo/factoryguard-ai.git zaalima_project
```

### Option C: Manual Upload

Upload the project files to `/tmp/zaalima_project` on your VPS using your preferred method (FTP, file manager, etc.).

## Step 3: Application Setup

### 3.1 Move Files to Application Directory

```bash
# Move project files to application directory
sudo mv /tmp/zaalima_project/* /var/www/html/factoryguard/
sudo mv /tmp/zaalima_project/.* /var/www/html/factoryguard/ 2>/dev/null || true

# Set proper permissions
sudo chown -R www-data:www-data /var/www/html/factoryguard
sudo chmod -R 755 /var/www/html/factoryguard
```

### 3.2 Create Python Virtual Environment

```bash
# Switch to application directory
cd /var/www/html/factoryguard

# Create virtual environment
sudo -u www-data python3 -m venv venv

# Activate virtual environment and install dependencies
sudo -u www-data bash -c "source venv/bin/activate && pip install --upgrade pip"
sudo -u www-data bash -c "source venv/bin/activate && pip install -r requirements.txt"
```

### 3.3 Test Application Locally

```bash
# Test if the application starts (as www-data user)
sudo -u www-data bash -c "cd /var/www/html/factoryguard && source venv/bin/activate && python app.py" &
sleep 5

# Test health endpoint
curl http://localhost:5000/health

# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 75.5, "vibration": 0.8, "pressure": 105.2}'

# Kill the test process
pkill -f "python app.py"
```

## Step 4: Configure Gunicorn

### 4.1 Create Gunicorn Configuration

```bash
# Create gunicorn configuration file
sudo tee /var/www/html/factoryguard/gunicorn.conf.py > /dev/null <<EOF
# Gunicorn configuration for FactoryGuard AI

bind = "127.0.0.1:8000"
workers = 3
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "/var/log/factoryguard/access.log"
errorlog = "/var/log/factoryguard/error.log"

# Process naming
proc_name = "factoryguard"

# Server mechanics
preload_app = True
pidfile = "/var/run/factoryguard/gunicorn.pid"
user = "www-data"
group = "www-data"
tmp_upload_dir = None

# Application
wsgi_module = "app:app"
EOF
```

### 4.2 Create Log Directory

```bash
# Create log directory
sudo mkdir -p /var/log/factoryguard
sudo chown www-data:www-data /var/log/factoryguard
```

### 4.3 Test Gunicorn

```bash
# Test gunicorn startup
cd /var/www/html/factoryguard
sudo -u www-data bash -c "source venv/bin/activate && gunicorn --config gunicorn.conf.py app:app"

# In another terminal, test the endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 75.5, "vibration": 0.8, "pressure": 105.2}'

# Stop gunicorn (Ctrl+C)
```

## Step 5: Configure Nginx

### 5.1 Create Nginx Configuration

```bash
# Create nginx site configuration
sudo tee /etc/nginx/sites-available/factoryguard > /dev/null <<EOF
server {
    listen 80;
    server_name your-vps-ip-or-domain;  # Replace with your VPS IP or domain

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Proxy to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Static files (if any)
    location /static/ {
        alias /var/www/html/factoryguard/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint (optional - for monitoring)
    location /nginx-health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF
```

### 5.2 Enable Site and Restart Nginx

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/factoryguard /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## Step 6: Create Systemd Service

### 6.1 Create Service File

```bash
# Create systemd service file
sudo tee /etc/systemd/system/factoryguard.service > /dev/null <<EOF
[Unit]
Description=FactoryGuard AI Predictive Maintenance API
After=network.target
Requires=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/var/www/html/factoryguard
Environment="PATH=/var/www/html/factoryguard/venv/bin"
ExecStart=/var/www/html/factoryguard/venv/bin/gunicorn --config gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
RestartSec=5
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

### 6.2 Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable factoryguard

# Start the service
sudo systemctl start factoryguard

# Check status
sudo systemctl status factoryguard
```

## Step 7: Testing and Verification

### 7.1 Test API Endpoints

```bash
# Test health endpoint
curl http://your-vps-ip/health

# Test prediction endpoint
curl -X POST http://your-vps-ip/predict \
     -H "Content-Type: application/json" \
     -d '{"timestamp": "2024-01-01T12:00:00", "temperature": 75.5, "vibration": 0.8, "pressure": 105.2}'

# Expected response format:
# {"failure_probability": 0.0234, "decision": "SAFE", "latency_ms": 15.67}
```

### 7.2 Test Latency Requirements

```bash
# Run latency test (if you have the test script)
cd /var/www/html/factoryguard
sudo -u www-data bash -c "source venv/bin/activate && python test_api_latency.py --url http://localhost:8000"
```

### 7.3 Monitor Logs

```bash
# View application logs
sudo journalctl -u factoryguard -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# View application-specific logs
sudo tail -f /var/log/factoryguard/access.log
sudo tail -f /var/log/factoryguard/error.log
```

## Step 8: Security Hardening

### 8.1 SSL/TLS Configuration (Optional but Recommended)

```bash
# Install certbot for Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com

# Test SSL
curl -I https://your-domain.com/health
```

### 8.2 Additional Security Measures

```bash
# Disable root login via SSH
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Install fail2ban for SSH protection
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Set up log rotation
sudo tee /etc/logrotate.d/factoryguard > /dev/null <<EOF
/var/log/factoryguard/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload factoryguard
    endscript
}
EOF
```

## Troubleshooting

### Common Issues

#### 1. Application Won't Start
```bash
# Check service status
sudo systemctl status factoryguard

# Check logs
sudo journalctl -u factoryguard -n 50

# Check if models are loaded
curl http://localhost:8000/health
```

#### 2. 502 Bad Gateway
```bash
# Check if gunicorn is running
ps aux | grep gunicorn

# Check nginx error log
sudo tail -f /var/log/nginx/error.log

# Restart services
sudo systemctl restart factoryguard
sudo systemctl restart nginx
```

#### 3. Permission Errors
```bash
# Fix permissions
sudo chown -R www-data:www-data /var/www/html/factoryguard
sudo chmod -R 755 /var/www/html/factoryguard
```

#### 4. Model Loading Errors
```bash
# Check if model files exist
ls -la /var/www/html/factoryguard/models/

# Test model loading manually
cd /var/www/html/factoryguard
sudo -u www-data bash -c "source venv/bin/activate && python -c \"import joblib; print(joblib.load('models/production_xgboost.joblib').keys())\""
```

#### 5. Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Adjust gunicorn workers
# Edit gunicorn.conf.py and reduce workers
workers = 2  # or 1
```

### Performance Tuning

#### Gunicorn Optimization
```python
# Adjust based on your VPS specs
# For 2GB RAM VPS:
workers = 2
worker_connections = 500

# For 4GB+ RAM VPS:
workers = 4
worker_connections = 1000
```

#### Nginx Optimization
```nginx
# Add to nginx config under server block
client_max_body_size 50M;
client_body_timeout 30;
```

## Monitoring and Maintenance

### Health Checks
```bash
# Create a simple monitoring script
sudo tee /usr/local/bin/check_factoryguard.sh > /dev/null <<EOF
#!/bin/bash
# Health check script for FactoryGuard AI

HEALTH_URL="http://localhost:8000/health"
PREDICT_URL="http://localhost:8000/predict"
TEST_DATA='{"timestamp": "2024-01-01T12:00:00", "temperature": 75.0, "vibration": 0.5, "pressure": 100.0}'

echo "Checking FactoryGuard AI health..."

# Check health endpoint
if curl -s -f \$HEALTH_URL > /dev/null; then
    echo "✓ Health endpoint OK"
else
    echo "✗ Health endpoint FAILED"
    exit 1
fi

# Check prediction endpoint
if curl -s -f -X POST -H "Content-Type: application/json" -d "\$TEST_DATA" \$PREDICT_URL > /dev/null; then
    echo "✓ Prediction endpoint OK"
else
    echo "✗ Prediction endpoint FAILED"
    exit 1
fi

echo "All checks passed!"
EOF

sudo chmod +x /usr/local/bin/check_factoryguard.sh
```

### Log Rotation
Already configured in security section above.

### Backup Strategy
```bash
# Create backup script
sudo tee /usr/local/bin/backup_factoryguard.sh > /dev/null <<EOF
#!/bin/bash
# Backup script for FactoryGuard AI

BACKUP_DIR="/var/backups/factoryguard"
DATE=\$(date +%Y%m%d_%H%M%S)

mkdir -p \$BACKUP_DIR

# Backup application files
tar -czf \$BACKUP_DIR/factoryguard_app_\$DATE.tar.gz -C /var/www/html factoryguard

# Backup models (most important)
tar -czf \$BACKUP_DIR/factoryguard_models_\$DATE.tar.gz -C /var/www/html/factoryguard models

# Clean old backups (keep last 7 days)
find \$BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: \$DATE"
EOF

sudo chmod +x /usr/local/bin/backup_factoryguard.sh

# Add to cron for daily backups
echo "0 2 * * * /usr/local/bin/backup_factoryguard.sh" | sudo crontab -
```

## API Documentation

### Endpoints

#### GET /health
Returns the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "pipeline_loaded": true
}
```

#### POST /predict
Makes a failure prediction based on sensor data.

**Request:**
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "temperature": 75.5,
  "vibration": 0.8,
  "pressure": 105.2
}
```

**Response:**
```json
{
  "failure_probability": 0.0234,
  "decision": "SAFE",
  "latency_ms": 15.67
}
```

### Error Responses

#### 400 Bad Request
```json
{
  "error": "Missing required field: temperature"
}
```

#### 500 Internal Server Error
```json
{
  "error": "Internal server error"
}
```

## Final Verification

After completing all steps, verify the deployment:

1. **API Accessibility:** `curl http://your-vps-ip/health`
2. **Prediction Functionality:** Test with sample data
3. **Latency:** Should be <50ms for predictions
4. **Logs:** Check that logs are being written
5. **Auto-start:** Reboot VPS and verify service starts automatically
6. **Security:** Ensure proper permissions and firewall rules

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review service and nginx logs
3. Verify all dependencies are installed
4. Test components individually (gunicorn, nginx, python app)

---

**FactoryGuard AI** - Production-Ready Predictive Maintenance API
