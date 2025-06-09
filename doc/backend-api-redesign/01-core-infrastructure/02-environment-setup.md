# Environment Setup - FaceSocial AI Services

## Development and Production Environment Configuration

This document provides comprehensive guidance for setting up development and production environments for the FaceSocial AI Services system.

---

## 🛠️ System Prerequisites

### Hardware Requirements

```yaml
Minimum Development Setup:
  GPU: RTX 3060 Laptop (6GB VRAM) or equivalent
  CPU: Intel i7-10750H / AMD Ryzen 7 4700H or better
  RAM: 16GB DDR4
  Storage: 100GB SSD available space
  Network: 100Mbps+ internet connection

Recommended Production Setup:
  GPU: RTX 4060/4070 (8GB+ VRAM)
  CPU: Intel i7-12700H / AMD Ryzen 7 6700H or better
  RAM: 32GB DDR5
  Storage: 500GB NVMe SSD
  Network: 1Gbps+ internet connection
```

### Software Prerequisites

```yaml
Operating System:
  Development: Windows 11, Ubuntu 22.04 LTS, macOS 12+ (with external GPU)
  Production: Ubuntu 22.04 LTS (recommended)

NVIDIA Drivers:
  Version: 525.60.11 or newer
  CUDA: 12.0+ support required

Container Runtime:
  Docker: 24.0+ with NVIDIA Container Toolkit
  Docker Compose: 2.20+

Development Tools:
  Git: 2.40+
  Python: 3.11+ (for local development)
  Visual Studio Code with extensions
```

---

## 🔧 Development Environment Setup

### 1. NVIDIA Driver and CUDA Installation

#### Windows Setup
```powershell
# Download and install NVIDIA Game Ready Driver
# https://www.nvidia.com/Download/index.aspx

# Download and install CUDA Toolkit 12.9
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi
nvcc --version
```

#### Ubuntu Setup
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install -y nvidia-driver-525

# Install CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-9

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvidia-smi
nvcc --version
```

### 2. Docker and NVIDIA Container Toolkit

#### Windows Docker Desktop
```powershell
# Install Docker Desktop with WSL 2 backend
# https://docs.docker.com/desktop/install/windows/

# Enable NVIDIA GPU support in Docker Desktop
# Settings > Resources > WSL Integration > Enable GPU support

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.9.0-base nvidia-smi
```

#### Ubuntu Docker Installation
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.9.0-base nvidia-smi
```

### 3. Project Setup

```bash
# Clone repository
git clone https://github.com/yourusername/facesocial-ai-services.git
cd facesocial-ai-services

# Create development environment file
cp .env.example .env.development

# Edit environment variables
nano .env.development
```

#### Development Environment Variables (`.env.development`)

```bash
# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug
API_HOST=0.0.0.0
API_PORT=8000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model Configuration
MODEL_PATH=/app/models
MODEL_CACHE_SIZE=1000
VRAM_ALLOCATION_STRATEGY=dynamic

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/facesocial_dev
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# Rate Limiting (Development - More Lenient)
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# File Upload Limits
MAX_IMAGE_SIZE=10485760  # 10MB
MAX_VIDEO_SIZE=104857600  # 100MB
ALLOWED_IMAGE_TYPES=image/jpeg,image/png,image/webp
ALLOWED_VIDEO_TYPES=video/mp4,video/webm

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000

# Development Tools
HOT_RELOAD=true
AUTO_RESTART=true
PROFILING_ENABLED=true
```

### 4. Model Download and Setup

```bash
# Create models directory structure
mkdir -p models/{face-detection,face-recognition,anti-spoofing,deepfake-detection,gender-age}

# Download models (example script)
./scripts/download-models.sh
```

#### Model Download Script (`scripts/download-models.sh`)

```bash
#!/bin/bash
set -e

MODELS_DIR="./models"
BASE_URL="https://your-model-storage.com"

echo "Downloading AI models..."

# Face Detection Models
echo "Downloading face detection models..."
wget -O "${MODELS_DIR}/face-detection/yolov10n-face.onnx" \
    "${BASE_URL}/face-detection/yolov10n-face.onnx"
wget -O "${MODELS_DIR}/face-detection/yolov5s-face.onnx" \
    "${BASE_URL}/face-detection/yolov5s-face.onnx"

# Face Recognition Models
echo "Downloading face recognition models..."
wget -O "${MODELS_DIR}/face-recognition/facenet_vggface2.onnx" \
    "${BASE_URL}/face-recognition/facenet_vggface2.onnx"
wget -O "${MODELS_DIR}/face-recognition/arcface_r100.onnx" \
    "${BASE_URL}/face-recognition/arcface_r100.onnx"
wget -O "${MODELS_DIR}/face-recognition/adaface_ir101.onnx" \
    "${BASE_URL}/face-recognition/adaface_ir101.onnx"

# Anti-Spoofing Models
echo "Downloading anti-spoofing models..."
wget -O "${MODELS_DIR}/anti-spoofing/AntiSpoofing_bin_1.5_128.onnx" \
    "${BASE_URL}/anti-spoofing/AntiSpoofing_bin_1.5_128.onnx"
wget -O "${MODELS_DIR}/anti-spoofing/AntiSpoofing_print-replay_1.5_128.onnx" \
    "${BASE_URL}/anti-spoofing/AntiSpoofing_print-replay_1.5_128.onnx"

# Deepfake Detection Model
echo "Downloading deepfake detection model..."
wget -O "${MODELS_DIR}/deepfake-detection/model.onnx" \
    "${BASE_URL}/deepfake-detection/model.onnx"

# Gender-Age Model
echo "Downloading gender-age model..."
wget -O "${MODELS_DIR}/gender-age/genderage.onnx" \
    "${BASE_URL}/gender-age/genderage.onnx"

# Verify downloads
echo "Verifying model integrity..."
sha256sum -c models/checksums.sha256

echo "Model download completed successfully!"
```

### 5. VS Code Development Setup

#### Required Extensions

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        "ms-vscode.vscode-json",
        "ms-azuretools.vscode-docker",
        "ms-vscode-remote.remote-containers",
        "github.copilot",
        "github.copilot-chat",
        "ms-python.debugpy"
    ]
}
```

#### VS Code Settings (`.vscode/settings.json`)

```json
{
    "python.defaultInterpreterPath": "/usr/bin/python3.11",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/node_modules": true,
        ".git": true,
        ".pytest_cache": true,
        ".mypy_cache": true
    },
    "docker.containers.label": "facesocial-ai",
    "remote.containers.defaultExtensions": [
        "ms-python.python",
        "ms-python.black-formatter",
        "charliermarsh.ruff"
    ]
}
```

#### Debug Configuration (`.vscode/launch.json`)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug FastAPI",
            "type": "python",
            "request": "launch",
            "program": "-m",
            "args": ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env.development",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v", "--tb=short"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env.development",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Debug Docker Container",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}/src",
                    "remoteRoot": "/app/src"
                }
            ]
        }
    ]
}
```

---

## 🏭 Production Environment Setup

### 1. Server Provisioning

#### Ubuntu Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    curl \
    wget \
    git \
    htop \
    nvidia-utils-525 \
    build-essential

# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Create application user
sudo useradd -m -s /bin/bash facesocial
sudo usermod -aG docker facesocial
```

#### Production Environment Variables (`.env.production`)

```bash
# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
API_HOST=0.0.0.0
API_PORT=8000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Model Configuration
MODEL_PATH=/app/models
MODEL_CACHE_SIZE=2000
VRAM_ALLOCATION_STRATEGY=production

# Database Configuration (Use secrets in production)
DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@db:5432/facesocial_prod
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0

# Security (Use strong secrets)
SECRET_KEY=${SECRET_KEY}
JWT_ALGORITHM=RS256
JWT_EXPIRE_MINUTES=30
JWT_REFRESH_EXPIRE_DAYS=7

# Rate Limiting (Production - Strict)
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
RATE_LIMIT_BURST=20

# File Upload Limits
MAX_IMAGE_SIZE=10485760  # 10MB
MAX_VIDEO_SIZE=52428800  # 50MB
ALLOWED_IMAGE_TYPES=image/jpeg,image/png
ALLOWED_VIDEO_TYPES=video/mp4

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000
SENTRY_DSN=${SENTRY_DSN}

# Performance
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000
MAX_CONCURRENT_REQUESTS=100

# SSL/TLS
SSL_ENABLED=true
SSL_CERT_PATH=/etc/ssl/certs/facesocial.crt
SSL_KEY_PATH=/etc/ssl/private/facesocial.key
```

### 2. SSL Certificate Setup

```bash
# Install Certbot for Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com -d api.your-domain.com

# Set up auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. Production Deployment

```bash
# Clone repository
cd /opt
sudo git clone https://github.com/yourusername/facesocial-ai-services.git
sudo chown -R facesocial:facesocial facesocial-ai-services
cd facesocial-ai-services

# Set up environment
sudo -u facesocial cp .env.example .env.production
sudo -u facesocial nano .env.production

# Download models
sudo -u facesocial ./scripts/download-models.sh

# Build and deploy
sudo -u facesocial ./scripts/deploy.sh production
```

---

## 🔒 Security Configuration

### 1. Secrets Management

#### Docker Secrets Setup
```bash
# Create secrets directory
mkdir -p secrets/

# Generate strong passwords
openssl rand -base64 32 > secrets/db_password.txt
openssl rand -base64 32 > secrets/redis_password.txt
openssl rand -base64 64 > secrets/secret_key.txt

# Generate JWT RSA keys
openssl genrsa -out secrets/jwt_private.pem 2048
openssl rsa -in secrets/jwt_private.pem -pubout -out secrets/jwt_public.pem

# Set proper permissions
chmod 600 secrets/*
chown root:root secrets/*
```

#### Environment Secrets Integration
```bash
# Create secure environment loader
cat > scripts/load-secrets.sh << 'EOF'
#!/bin/bash
export DB_PASSWORD=$(cat secrets/db_password.txt)
export REDIS_PASSWORD=$(cat secrets/redis_password.txt)
export SECRET_KEY=$(cat secrets/secret_key.txt)
export JWT_PRIVATE_KEY_PATH=secrets/jwt_private.pem
export JWT_PUBLIC_KEY_PATH=secrets/jwt_public.pem
EOF

chmod +x scripts/load-secrets.sh
```

### 2. Network Security

#### Firewall Configuration
```bash
# Configure UFW for production
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432  # Database access
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis access
sudo ufw enable
```

#### Nginx Security Headers
```nginx
# Add to nginx.conf
server {
    # ... existing configuration ...
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

---

## 📊 Monitoring Setup

### 1. System Monitoring

#### Prometheus Configuration (`monitoring/prometheus.yml`)
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'ai-services'
    static_configs:
      - targets: ['ai-services:8000']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['gpu-exporter:9445']
```

#### GPU Monitoring Setup
```bash
# Install NVIDIA GPU exporter
docker run -d \
  --name gpu-exporter \
  --restart unless-stopped \
  --gpus all \
  -p 9445:9445 \
  mindprince/nvidia_gpu_prometheus_exporter:0.1
```

### 2. Log Management

#### Structured Logging Configuration
```python
# src/core/logging.py
import structlog
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/app/logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    },
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

---

## 🧪 Testing Environment

### 1. Test Configuration

```bash
# Test environment variables (.env.test)
ENVIRONMENT=test
DEBUG=true
DATABASE_URL=sqlite:///./test.db
REDIS_URL=redis://localhost:6379/1
SECRET_KEY=test-secret-key
MODEL_PATH=./tests/fixtures/models
SKIP_GPU_TESTS=true  # For CI/CD environments without GPU
```

### 2. Test Data Setup

```bash
# Create test fixtures
mkdir -p tests/fixtures/{images,videos,models}

# Download test models (smaller versions)
./scripts/download-test-models.sh
```

### 3. Continuous Integration

#### GitHub Actions Configuration (`.github/workflows/ci.yml`)
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/test.txt
    
    - name: Run tests
      env:
        SKIP_GPU_TESTS: true
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t facesocial-ai:test .
```

---

*This environment setup provides a complete foundation for developing, testing, and deploying the FaceSocial AI Services system across different environments.*
