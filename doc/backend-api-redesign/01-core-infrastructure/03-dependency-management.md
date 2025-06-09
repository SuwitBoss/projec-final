# Dependency Management

Comprehensive dependency management for the VRAM-optimized AI backend system including Python packages, CUDA libraries, system dependencies, and development tools.

## 📦 Dependency Overview

### Core Technology Stack
- **Python**: 3.10+ (Primary runtime)
- **CUDA**: 12.9.0 (GPU acceleration)
- **cuDNN**: 9.10 (Deep learning optimization)
- **Docker**: 24.0+ (Containerization)
- **Redis**: 7.0+ (Caching and queues)
- **PostgreSQL**: 15+ (Database)

### Package Management Strategy
- **Production**: Pinned versions for stability
- **Development**: Range versions for flexibility
- **Security**: Regular dependency scanning
- **Performance**: Optimized builds and caching

## 🐍 Python Dependencies

### Core Requirements (`requirements.txt`)

```txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# ASGI & WebSocket
uvloop==0.19.0
websockets==12.0

# Database & Caching
asyncpg==0.29.0
redis==5.0.1
sqlalchemy[asyncio]==2.0.23
alembic==1.13.1

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
cryptography==41.0.8

# AI/ML Core Libraries
torch==2.1.1+cu121
torchvision==0.16.1+cu121
torchaudio==2.1.1+cu121
numpy==1.24.4
opencv-python==4.8.1.78
Pillow==10.1.0
scikit-learn==1.3.2

# Deep Learning Frameworks
transformers==4.36.0
timm==0.9.12
onnxruntime-gpu==1.16.3
tensorrt==8.6.1

# Face Analysis Libraries
face-recognition==1.3.0
mediapipe==0.10.8
insightface==0.7.3
dlib==19.24.2

# Computer Vision
albumentations==1.3.1
imgaug==0.4.0
scikit-image==0.22.0

# HTTP & API
httpx==0.25.2
aiofiles==23.2.1
python-magic==0.4.27

# Monitoring & Logging
prometheus-client==0.19.0
structlog==23.2.0
sentry-sdk[fastapi]==1.38.0

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1

# Performance & Optimization
cython==3.0.6
numba==0.58.1
```

### AI Model Specific Requirements (`requirements-models.txt`)

```txt
# YOLO Dependencies
ultralytics==8.0.224
yolov5==7.0.13

# Face Recognition Models
facenet-pytorch==2.5.3
arcface-torch==0.0.1

# Anti-spoofing Models
silent-face-anti-spoofing==1.0.0

# Deepfake Detection
efficientnet-pytorch==0.7.1
pretrainedmodels==0.7.4

# Age/Gender Detection
tensorflow==2.15.0  # For some pre-trained models
keras==2.15.0

# Model Optimization
torch-tensorrt==1.4.0
onnx==1.15.0
onnxsim==0.4.35
```

### Development Requirements (`requirements-dev.txt`)

```txt
# Code Quality
pre-commit==3.6.0
ruff==0.1.6
bandit==1.7.5
safety==2.3.4

# Documentation
mkdocs==1.5.3
mkdocs-material==9.4.8
mkdocstrings[python]==0.24.0

# API Documentation
fastapi-users==12.1.2
redoc-cli==0.13.2

# Performance Testing
locust==2.17.0
memory-profiler==0.61.0
line-profiler==4.1.1

# Database Tools
pgcli==4.0.1
redis-cli==3.5.3

# Container Tools
docker-compose==1.29.2
```

## 🐳 Docker Dependencies

### Base Image Selection

```dockerfile
# Multi-stage build for optimized production image
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04 as builder

# System dependencies for building
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    pkg-config \
    libhdf5-dev \
    libopencv-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Python build environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies
COPY requirements*.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-models.txt

# Production image
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

# Runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libhdf5-103 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
```

### Docker Compose Dependencies

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    depends_on:
      - redis
      - postgres
      - nginx
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7.2-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 1gb

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: facesocial
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:1.25-alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

volumes:
  redis_data:
  postgres_data:
```

## 🛠️ System Dependencies

### Ubuntu 22.04 System Packages

```bash
#!/bin/bash
# system-deps.sh - Install system dependencies

# Update package lists
apt-get update

# Build tools and compilers
apt-get install -y \
    build-essential \
    cmake \
    gcc-11 \
    g++-11 \
    gfortran \
    pkg-config \
    git \
    curl \
    wget

# Python development
apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    python3-setuptools \
    python3-wheel

# Graphics and media libraries
apt-get install -y \
    libopencv-dev \
    libopencv-contrib-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev

# Linear algebra libraries
apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libeigen3-dev

# Database clients
apt-get install -y \
    postgresql-client \
    redis-tools

# Monitoring tools
apt-get install -y \
    htop \
    iotop \
    nvidia-smi \
    nvtop

# Clean up
apt-get autoremove -y
apt-get autoclean
rm -rf /var/lib/apt/lists/*
```

### NVIDIA Dependencies

```bash
#!/bin/bash
# nvidia-setup.sh - Install NVIDIA dependencies

# NVIDIA Driver (if not already installed)
# ubuntu-drivers autoinstall

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Verify CUDA installation
nvidia-smi
nvcc --version
```

## 📋 Dependency Pinning Strategy

### Production Pinning (`requirements-prod.txt`)

```txt
# Exact versions for production stability
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.1+cu121
torchvision==0.16.1+cu121
numpy==1.24.4
opencv-python==4.8.1.78
redis==5.0.1
asyncpg==0.29.0
pydantic==2.5.0
transformers==4.36.0
```

### Development Ranges (`requirements-dev-ranges.txt`)

```txt
# Compatible ranges for development flexibility
fastapi>=0.104.0,<0.105.0
uvicorn>=0.24.0,<0.25.0
torch>=2.1.0,<2.2.0
torchvision>=0.16.0,<0.17.0
numpy>=1.24.0,<1.25.0
opencv-python>=4.8.0,<4.9.0
redis>=5.0.0,<5.1.0
asyncpg>=0.29.0,<0.30.0
```

## 🔐 Security Dependencies

### Security Scanning Tools

```txt
# Security scanning in CI/CD
safety==2.3.4          # Known vulnerability scanning
bandit==1.7.5           # Python security linting
semgrep==1.45.0         # Static analysis security
pip-audit==2.6.1        # Audit pip packages
cyclonedx-bom==4.1.0    # Software bill of materials
```

### Security Configuration

```python
# security_deps.py - Security dependency management
import subprocess
import json
from typing import List, Dict

def scan_vulnerabilities() -> Dict:
    """Scan for known vulnerabilities in dependencies."""
    result = subprocess.run([
        'safety', 'check', '--json'
    ], capture_output=True, text=True)
    
    return json.loads(result.stdout) if result.stdout else {}

def audit_packages() -> List[Dict]:
    """Audit packages for security issues."""
    result = subprocess.run([
        'pip-audit', '--format=json'
    ], capture_output=True, text=True)
    
    return json.loads(result.stdout) if result.stdout else []

def generate_sbom() -> str:
    """Generate Software Bill of Materials."""
    subprocess.run([
        'cyclonedx-py', '--output-format', 'json', 
        '--output-file', 'sbom.json'
    ])
    return 'sbom.json'
```

## 🚀 Performance Dependencies

### Optimized Builds

```bash
#!/bin/bash
# build-optimized.sh - Build optimized dependencies

# Compile optimized NumPy with OpenBLAS
export BLAS=openblas
export LAPACK=openblas
pip install --no-binary=numpy numpy

# Compile OpenCV with optimizations
export CMAKE_ARGS="-DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_FAST_MATH=ON"
pip install --no-binary=opencv-python opencv-python

# Install PyTorch with CUDA optimizations
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Compile optimized TensorRT
pip install tensorrt --find-links https://developer.nvidia.com/tensorrt
```

### Performance Monitoring

```python
# perf_deps.py - Performance dependency monitoring
import psutil
import GPUtil
import time
from typing import Dict

def monitor_dependencies() -> Dict:
    """Monitor dependency performance impact."""
    
    # GPU memory usage
    gpus = GPUtil.getGPUs()
    gpu_memory = gpus[0].memoryUsed if gpus else 0
    
    # System memory
    memory = psutil.virtual_memory()
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        'gpu_memory_mb': gpu_memory,
        'ram_usage_percent': memory.percent,
        'cpu_usage_percent': cpu_percent,
        'timestamp': time.time()
    }

def profile_import_times():
    """Profile import times for heavy dependencies."""
    import time
    
    heavy_imports = [
        'torch', 'torchvision', 'transformers',
        'opencv', 'numpy', 'mediapipe'
    ]
    
    import_times = {}
    for module in heavy_imports:
        start = time.time()
        try:
            __import__(module)
            import_times[module] = time.time() - start
        except ImportError:
            import_times[module] = None
    
    return import_times
```

## 📊 Dependency Management Tools

### Automated Updates

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "ai-team"
    assignees:
      - "devops-team"
    allow:
      - dependency-type: "direct"
      - dependency-type: "indirect"
    ignore:
      - dependency-name: "torch*"
        update-types: ["version-update:semver-major"]
```

### CI/CD Integration

```yaml
# .github/workflows/dependencies.yml
name: Dependency Management

on:
  push:
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Security Scan
        run: |
          pip install safety bandit
          safety check -r requirements.txt
          bandit -r src/
  
  dependency-update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Update Dependencies
        run: |
          pip-compile requirements.in
          pip-compile requirements-dev.in
          
  vulnerability-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Vulnerability Check
        run: |
          pip install pip-audit
          pip-audit --format=json --output=vuln-report.json
```

## 🔧 Virtual Environment Management

### Development Environment Setup

```bash
#!/bin/bash
# setup-dev-env.sh - Setup development environment

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip and tools
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -r requirements-dev.txt
pip install -r requirements.txt
pip install -r requirements-models.txt

# Install pre-commit hooks
pre-commit install

# Setup environment variables
cp .env.example .env
echo "Development environment ready!"
```

### Production Environment

```bash
#!/bin/bash
# setup-prod-env.sh - Setup production environment

# Create optimized virtual environment
python3.10 -m venv --system-site-packages venv
source venv/bin/activate

# Install only production dependencies
pip install --no-deps -r requirements-prod.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo "Production environment ready!"
```

## 📈 Dependency Monitoring

### Health Checks

```python
# deps_health.py - Dependency health monitoring
import importlib
import sys
from typing import Dict, List

CRITICAL_DEPS = [
    'fastapi', 'torch', 'torchvision', 'opencv',
    'numpy', 'redis', 'asyncpg', 'transformers'
]

def check_dependency_health() -> Dict:
    """Check health of critical dependencies."""
    health_status = {
        'healthy': [],
        'missing': [],
        'outdated': [],
        'import_errors': []
    }
    
    for dep in CRITICAL_DEPS:
        try:
            module = importlib.import_module(dep)
            health_status['healthy'].append({
                'name': dep,
                'version': getattr(module, '__version__', 'unknown')
            })
        except ImportError as e:
            health_status['import_errors'].append({
                'name': dep,
                'error': str(e)
            })
    
    return health_status

def get_gpu_dependencies() -> Dict:
    """Check GPU-specific dependencies."""
    gpu_deps = {}
    
    try:
        import torch
        gpu_deps['torch_cuda'] = torch.cuda.is_available()
        gpu_deps['cuda_version'] = torch.version.cuda
    except ImportError:
        gpu_deps['torch_cuda'] = False
    
    try:
        import cv2
        gpu_deps['opencv_cuda'] = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        gpu_deps['opencv_cuda'] = False
    
    return gpu_deps
```

---

*Next: [02-vram-management/02-model-lifecycle-management.md](../02-vram-management/02-model-lifecycle-management.md) - Model Lifecycle Management*
