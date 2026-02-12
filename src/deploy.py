import docker
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def create_dockerfile(project_root: Path) -> str:
    """
    Create Dockerfile for the predictive maintenance API.

    Args:
        project_root: Path to project root

    Returns:
        Dockerfile content as string
    """
    dockerfile_content = f'''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "-m", "src.api"]
'''

    return dockerfile_content

def create_docker_compose(project_root: Path) -> str:
    """
    Create docker-compose.yml for the application.

    Args:
        project_root: Path to project root

    Returns:
        docker-compose content as string
    """
    compose_content = '''version: '3.8'

services:
  factoryguard-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
'''

    return compose_content

def build_docker_image(project_root: Path, image_name: str = "factoryguard-ai:latest") -> bool:
    """
    Build Docker image for the application.

    Args:
        project_root: Path to project root
        image_name: Name for the Docker image

    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()

        # Build the image
        logger.info(f"Building Docker image: {image_name}")
        image, build_logs = client.images.build(
            path=str(project_root),
            tag=image_name,
            rm=True
        )

        # Print build logs
        for log in build_logs:
            if 'stream' in log:
                print(log['stream'].strip())

        logger.info(f"Successfully built Docker image: {image_name}")
        return True

    except docker.errors.BuildError as e:
        logger.error(f"Docker build failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error building Docker image: {e}")
        return False

def deploy_locally(project_root: Path) -> bool:
    """
    Deploy the application locally using Docker Compose.

    Args:
        project_root: Path to project root

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess

        # Check if docker-compose is available
        result = subprocess.run(['docker-compose', '--version'],
                              capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("docker-compose not found. Please install Docker Compose.")
            return False

        # Run docker-compose up
        logger.info("Starting application with Docker Compose...")
        os.chdir(project_root)

        result = subprocess.run(['docker-compose', 'up', '-d'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Application deployed successfully!")
            logger.info("API available at: http://localhost:5000")
            logger.info("Health check: http://localhost:5000/health")
            return True
        else:
            logger.error(f"Docker Compose failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error deploying locally: {e}")
        return False

def create_deployment_package(project_root: Path, output_dir: str = "deployment") -> str:
    """
    Create deployment package with all necessary files.

    Args:
        project_root: Path to project root
        output_dir: Output directory for deployment package

    Returns:
        Path to created deployment package
    """
    try:
        deployment_path = project_root / output_dir
        deployment_path.mkdir(exist_ok=True)

        # Create Dockerfile
        dockerfile_content = create_dockerfile(project_root)
        with open(deployment_path / "Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Create docker-compose.yml
        compose_content = create_docker_compose(project_root)
        with open(deployment_path / "docker-compose.yml", "w") as f:
            f.write(compose_content)

        # Copy requirements.txt
        import shutil
        shutil.copy(project_root / "requirements.txt", deployment_path)

        # Copy source code
        shutil.copytree(project_root / "src", deployment_path / "src", dirs_exist_ok=True)

        # Copy models directory
        shutil.copytree(project_root / "models", deployment_path / "models", dirs_exist_ok=True)

        # Create deployment README
        readme_content = create_deployment_readme()
        with open(deployment_path / "README.md", "w") as f:
            f.write(readme_content)

        logger.info(f"Deployment package created at: {deployment_path}")
        return str(deployment_path)

    except Exception as e:
        logger.error(f"Error creating deployment package: {e}")
        return ""

def create_deployment_readme() -> str:
    """
    Create README for deployment package.

    Returns:
        README content as string
    """
    readme = '''# FactoryGuard AI - Deployment

## Quick Start

### Using Docker Compose (Recommended)

1. **Prerequisites:**
   - Docker installed and running
   - Docker Compose installed

2. **Deploy:**
   ```bash
   docker-compose up -d
   ```

3. **Check Status:**
   ```bash
   # Check if container is running
   docker-compose ps

   # Check application health
   curl http://localhost:5000/health
   ```

4. **Test Prediction:**
   ```bash
   curl -X POST http://localhost:5000/predict \\
        -H "Content-Type: application/json" \\
        -d '{
          "temperature": 75.5,
          "vibration": 0.8,
          "pressure": 105.2
        }'
   ```

### Manual Deployment

1. **Build Image:**
   ```bash
   docker build -t factoryguard-ai .
   ```

2. **Run Container:**
   ```bash
   docker run -p 5000:5000 -v $(pwd)/models:/app/models:ro factoryguard-ai
   ```

## API Endpoints

- **POST /predict**: Make failure prediction
- **GET /health**: Health check

## Configuration

The application loads models from the `models/` directory:
- `production_xgboost.joblib`: Trained XGBoost model
- `feature_pipeline.joblib`: Feature engineering pipeline

## Monitoring

- Health checks every 30 seconds
- Automatic restart on failure
- Logs available via `docker-compose logs`

## Troubleshooting

1. **Port already in use:**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "5001:5000"
   ```

2. **Model loading errors:**
   - Ensure models directory is mounted correctly
   - Check model files exist and are not corrupted

3. **Memory issues:**
   - Increase Docker memory limit
   - Use smaller batch sizes for predictions
'''

    return readme

def run_deployment_checks(project_root: Path) -> Dict[str, Any]:
    """
    Run pre-deployment checks.

    Args:
        project_root: Path to project root

    Returns:
        Dictionary with check results
    """
    checks = {
        'docker_installed': False,
        'docker_compose_installed': False,
        'models_exist': False,
        'requirements_exist': False,
        'source_code_exists': False
    }

    try:
        # Check Docker
        import subprocess
        result = subprocess.run(['docker', '--version'], capture_output=True)
        checks['docker_installed'] = result.returncode == 0

        # Check Docker Compose
        result = subprocess.run(['docker-compose', '--version'], capture_output=True)
        checks['docker_compose_installed'] = result.returncode == 0

        # Check required files
        checks['models_exist'] = (project_root / 'models').exists()
        checks['requirements_exist'] = (project_root / 'requirements.txt').exists()
        checks['source_code_exists'] = (project_root / 'src').exists()

    except Exception as e:
        logger.error(f"Error running deployment checks: {e}")

    return checks
