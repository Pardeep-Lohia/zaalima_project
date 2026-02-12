# FactoryGuard AI VPS Deployment TODO

## Pre-Deployment Checklist
- [ ] Access to blank Ubuntu 20.04 VPS with root SSH access
- [ ] Project files ready for transfer (app.py, requirements.txt, models/, src/)
- [ ] VPS IP address or domain name noted
- [ ] Local machine has SCP or file transfer capability

## VPS Initial Setup
- [ ] Update system packages (`sudo apt update && sudo apt upgrade -y`)
- [ ] Install required packages (python3, pip, nginx, git, ufw)
- [ ] Configure UFW firewall (allow SSH, HTTP, HTTPS)
- [ ] Create application directory (`/var/www/html/factoryguard`)

## File Transfer
- [ ] Transfer project files to VPS (`/tmp/zaalima_project`)
- [ ] Move files to application directory
- [ ] Set proper ownership and permissions (www-data:www-data)

## Python Environment Setup
- [ ] Create Python virtual environment
- [ ] Install Python dependencies from requirements.txt
- [ ] Test application startup locally

## Gunicorn Configuration
- [ ] Create gunicorn.conf.py configuration file
- [ ] Create log directory (/var/log/factoryguard)
- [ ] Test gunicorn startup and API endpoints

## Nginx Configuration
- [ ] Create nginx site configuration (/etc/nginx/sites-available/factoryguard)
- [ ] Enable site and remove default
- [ ] Test nginx configuration and restart service

## Systemd Service Setup
- [ ] Create factoryguard.service systemd file
- [ ] Enable and start the service
- [ ] Verify service status and auto-startup

## Testing and Verification
- [ ] Test health endpoint via public IP
- [ ] Test prediction endpoint with sample data
- [ ] Run latency tests (<50ms requirement)
- [ ] Verify model loading and functionality

## Security Hardening (Optional but Recommended)
- [ ] Configure SSL/TLS with Let's Encrypt
- [ ] Disable root SSH login
- [ ] Install and configure fail2ban
- [ ] Set up log rotation

## Monitoring and Maintenance
- [ ] Set up health check script
- [ ] Configure backup strategy
- [ ] Test log rotation
- [ ] Verify monitoring scripts

## Final Verification
- [ ] Confirm API is accessible publicly
- [ ] Validate prediction accuracy and latency
- [ ] Test service restart after reboot
- [ ] Review all logs for errors
- [ ] Document any custom configurations made
