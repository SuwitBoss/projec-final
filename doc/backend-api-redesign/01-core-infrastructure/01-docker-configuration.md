# Docker Configuration - FaceSocial AI Services

## Docker Environment for NVIDIA CUDA 12.9 + cuDNN 9.10

This document provides comprehensive Docker configuration for deploying FaceSocial AI Services with GPU acceleration on RTX 3060 Laptop hardware.

---

## 🐳 Base Image Strategy

### Multi-Stage Build Architecture

```dockerfile
# ============================================================================
# Stage 1: Base CUDA Environment
# ============================================================================
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04 as cuda-base

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VERSION=12.9
ENV CUDNN_VERSION=9.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog0v5 \
    libgflags2.2 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN ln -s /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Python Dependencies
# ============================================================================
FROM cuda-base as python-deps

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements/base.txt requirements/production.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r production.txt

# Install ONNX Runtime GPU
RUN pip install --no-cache-dir onnxruntime-gpu==1.16.3 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM python-deps as application

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory and permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/temp /app/cache && \
    chown -R appuser:appuser /app/logs /app/temp /app/cache

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 4: Development (Optional)
# ============================================================================
FROM python-deps as development

# Install development dependencies
COPY requirements/development.txt ./
RUN pip install --no-cache-dir -r development.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-mock \
    black \
    ruff \
    mypy \
    pre-commit

# Create non-root user for development
RUN groupadd -r devuser && useradd -r -g devuser devuser
WORKDIR /app
RUN chown -R devuser:devuser /app
USER devuser

# Development command with hot reload
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

---

## 📦 Requirements Management

### Base Requirements (`requirements/base.txt`)

```txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# ASGI Server
gunicorn==21.2.0

# HTTP Client
httpx==0.25.2
aiohttp==3.9.1

# Image Processing
opencv-python==4.8.1.78
Pillow==10.1.0
numpy==1.24.4
scipy==1.11.4

# AI/ML Libraries
torch==2.1.1+cu121
torchvision==0.16.1+cu121
onnx==1.15.0
# onnxruntime-gpu installed separately in Dockerfile

# Computer Vision
mediapipe==0.10.8
insightface==0.7.3

# Data Processing
pandas==2.1.3
scikit-learn==1.3.2

# Async Processing
celery==5.3.4
redis==5.0.1
asyncio-mqtt==0.13.0

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.13.1

# Caching
cachetools==5.3.2
diskcache==5.6.3

# Monitoring
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0

# Utilities
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
typer==0.9.0
rich==13.7.0

# Time handling
python-dateutil==2.8.2
pytz==2023.3
```

### Production Requirements (`requirements/production.txt`)

```txt
-r base.txt

# Production WSGI/ASGI
gunicorn==21.2.0
gevent==23.9.1

# Production Database
psycopg2-binary==2.9.9

# Production Monitoring
sentry-sdk[fastapi]==1.38.0
structlog==23.2.0

# Production Security
cryptography==41.0.7

# Performance
ujson==5.8.0
orjson==3.9.10
```

### Development Requirements (`requirements/development.txt`)

```txt
-r base.txt

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
httpx==0.25.2

# Code Quality
black==23.11.0
ruff==0.1.6
mypy==1.7.1
pre-commit==3.6.0

# Development Tools
ipython==8.17.2
jupyter==1.0.0
notebook==7.0.6

# Debugging
pdbpp==0.10.3
ipdb==0.13.13

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0
```

---

## 🔧 Docker Compose Configuration

### Development Environment (`docker-compose.dev.yml`)

```yaml
version: '3.8'

services:
  # Main AI Services Application
  ai-services:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    container_name: facesocial-ai-dev
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - CUDA_VISIBLE_DEVICES=0
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/facesocial_dev
      - MODEL_PATH=/app/models
      - LOG_LEVEL=debug
    volumes:
      - ./src:/app/src
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
      - model-cache:/app/cache
    depends_on:
      - redis
      - db
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - ai-network
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: facesocial-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    networks:
      - ai-network
    restart: unless-stopped

  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    container_name: facesocial-db-dev
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=facesocial_dev
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - ai-network
    restart: unless-stopped

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: facesocial-prometheus-dev
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - ai-network
    restart: unless-stopped

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:10.2.0
    container_name: facesocial-grafana-dev
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - ai-network
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  grafana-data:
  prometheus-data:
  model-cache:

networks:
  ai-network:
    driver: bridge
```

### Production Environment (`docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    container_name: facesocial-nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - ai-services
    networks:
      - ai-network
    restart: unless-stopped

  # AI Services (Production)
  ai-services:
    build:
      context: .
      dockerfile: Dockerfile
      target: application
    container_name: facesocial-ai-prod
    expose:
      - "8000"
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - CUDA_VISIBLE_DEVICES=0
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@db:5432/facesocial_prod
      - MODEL_PATH=/app/models
      - LOG_LEVEL=info
      - SENTRY_DSN=${SENTRY_DSN}
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - model-cache:/app/cache
    depends_on:
      - redis
      - db
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 8G
          cpus: '4'
    networks:
      - ai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/system/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Redis Cache (Production)
  redis:
    image: redis:7-alpine
    container_name: facesocial-redis-prod
    volumes:
      - redis-data:/data
      - ./redis/redis.conf:/etc/redis/redis.conf
    command: redis-server /etc/redis/redis.conf
    networks:
      - ai-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  # PostgreSQL Database (Production)
  db:
    image: postgres:15-alpine
    container_name: facesocial-db-prod
    environment:
      - POSTGRES_DB=facesocial_prod
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - ai-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

  # Backup Service
  backup:
    image: postgres:15-alpine
    container_name: facesocial-backup
    environment:
      - POSTGRES_DB=facesocial_prod
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - ./backups:/backups
      - ./scripts/backup.sh:/backup.sh
    command: >
      sh -c "
        chmod +x /backup.sh &&
        crond -f -l 8
      "
    depends_on:
      - db
    networks:
      - ai-network
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  model-cache:

networks:
  ai-network:
    driver: bridge

secrets:
  db_password:
    file: ./secrets/db_password.txt
  sentry_dsn:
    file: ./secrets/sentry_dsn.txt
```

---

## ⚙️ Configuration Files

### Nginx Configuration (`nginx/nginx.conf`)

```nginx
events {
    worker_connections 1024;
}

http {
    upstream ai_backend {
        server ai-services:8000;
        keepalive 32;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for" '
                   'rt=$request_time uct="$upstream_connect_time" '
                   'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss
               application/atom+xml image/svg+xml;

    server {
        listen 80;
        server_name localhost;
        client_max_body_size 100M;

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://ai_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts for AI processing
            proxy_connect_timeout 10s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # File upload endpoints with stricter rate limiting
        location ~* /api/v1/(face|video|batch)/ {
            limit_req zone=upload burst=5 nodelay;
            
            proxy_pass http://ai_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Extended timeouts for file processing
            proxy_connect_timeout 10s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
        }

        # WebSocket support
        location /ws/ {
            proxy_pass http://ai_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### Redis Configuration (`redis/redis.conf`)

```conf
# Basic Configuration
bind 0.0.0.0
port 6379
timeout 300
tcp-keepalive 60

# Memory Management
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dir /data

# Append Only File
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Logging
loglevel notice
logfile /data/redis.log

# Security (for production)
# requirepass your_redis_password_here

# Performance
hz 10
tcp-backlog 511
```

---

## 🚀 Build and Deployment Scripts

### Build Script (`scripts/build.sh`)

```bash
#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building FaceSocial AI Services Docker Images${NC}"

# Check if NVIDIA Docker is available
if ! command -v nvidia-docker &> /dev/null; then
    echo -e "${RED}NVIDIA Docker not found. Please install nvidia-container-toolkit${NC}"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA GPU not detected. Please check your NVIDIA drivers${NC}"
    exit 1
fi

# Build development image
echo -e "${YELLOW}Building development image...${NC}"
docker build --target development -t facesocial-ai:dev .

# Build production image
echo -e "${YELLOW}Building production image...${NC}"
docker build --target application -t facesocial-ai:latest .

# Tag images with version
VERSION=$(git describe --tags --always --dirty)
docker tag facesocial-ai:latest facesocial-ai:${VERSION}

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}Images created:${NC}"
echo -e "  - facesocial-ai:dev"
echo -e "  - facesocial-ai:latest"
echo -e "  - facesocial-ai:${VERSION}"
```

### Deployment Script (`scripts/deploy.sh`)

```bash
#!/bin/bash
set -e

# Configuration
ENVIRONMENT=${1:-development}
COMPOSE_FILE="docker-compose.${ENVIRONMENT}.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Deploying FaceSocial AI Services - ${ENVIRONMENT}${NC}"

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}Compose file $COMPOSE_FILE not found${NC}"
    exit 1
fi

# Load environment variables
if [ -f ".env.${ENVIRONMENT}" ]; then
    export $(cat .env.${ENVIRONMENT} | xargs)
fi

# Pre-deployment checks
echo -e "${YELLOW}Running pre-deployment checks...${NC}"

# Check NVIDIA runtime
if ! docker info | grep -q "nvidia"; then
    echo -e "${RED}NVIDIA Docker runtime not configured${NC}"
    exit 1
fi

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}GPU not available${NC}"
    exit 1
fi

# Check required environment variables for production
if [ "$ENVIRONMENT" = "production" ]; then
    required_vars=("DB_PASSWORD" "SENTRY_DSN")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo -e "${RED}Required environment variable $var is not set${NC}"
            exit 1
        fi
    done
fi

# Deploy
echo -e "${YELLOW}Starting deployment...${NC}"
docker-compose -f "$COMPOSE_FILE" down --remove-orphans
docker-compose -f "$COMPOSE_FILE" pull
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Health check
if [ "$ENVIRONMENT" = "production" ]; then
    HEALTH_URL="http://localhost/api/v1/system/health"
else
    HEALTH_URL="http://localhost:8000/api/v1/system/health"
fi

for i in {1..30}; do
    if curl -f "$HEALTH_URL" &> /dev/null; then
        echo -e "${GREEN}Services are healthy!${NC}"
        break
    fi
    echo "Waiting for services... ($i/30)"
    sleep 5
done

# Show running containers
echo -e "${GREEN}Deployment completed!${NC}"
docker-compose -f "$COMPOSE_FILE" ps
```

---

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

```yaml
CUDA/GPU Issues:
  Problem: "CUDA out of memory"
  Solution: 
    - Check VRAM usage with nvidia-smi
    - Restart container to clear GPU memory
    - Reduce batch size or model concurrency

  Problem: "NVIDIA runtime not found"
  Solution:
    - Install nvidia-container-toolkit
    - Restart Docker daemon
    - Verify with: docker run --gpus all nvidia/cuda:12.9.0-base nvidia-smi

Container Issues:
  Problem: "Container exits immediately"
  Solution:
    - Check logs: docker logs facesocial-ai-dev
    - Verify environment variables
    - Check file permissions

  Problem: "Port already in use"
  Solution:
    - Check running containers: docker ps
    - Kill conflicting processes: sudo lsof -i :8000
    - Use different ports in compose file

Model Loading Issues:
  Problem: "Model file not found"
  Solution:
    - Verify model path mapping in volumes
    - Check file permissions (readable by appuser)
    - Ensure model files are in correct format

Performance Issues:
  Problem: "Slow response times"
  Solution:
    - Monitor GPU utilization
    - Check CPU fallback usage
    - Verify network latency
    - Review memory allocation strategy
```

### Diagnostic Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check container logs
docker logs -f facesocial-ai-dev

# Execute commands in container
docker exec -it facesocial-ai-dev bash

# Check container resource usage
docker stats

# Inspect container configuration
docker inspect facesocial-ai-dev

# Test CUDA availability in container
docker exec facesocial-ai-dev python -c "import torch; print(torch.cuda.is_available())"

# Check ONNX Runtime providers
docker exec facesocial-ai-dev python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

---

*This Docker configuration provides a robust foundation for deploying FaceSocial AI Services with optimal GPU utilization and production-ready features.*
