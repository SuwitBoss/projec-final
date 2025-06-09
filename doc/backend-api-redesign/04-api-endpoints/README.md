# API Endpoints Overview

This section provides comprehensive documentation for all API endpoints in the VRAM-optimized AI backend system.

## 📚 Section Contents

### Core API Documentation
- **[01-authentication-api.md](01-authentication-api.md)** - Authentication & authorization endpoints
- **[02-face-analysis-api.md](02-face-analysis-api.md)** - Real-time face analysis endpoints
- **[03-batch-processing-api.md](03-batch-processing-api.md)** - Batch processing & queue management
- **[04-system-management-api.md](04-system-management-api.md)** - System control & monitoring endpoints
- **[05-websocket-realtime-api.md](05-websocket-realtime-api.md)** - WebSocket real-time communication

## 🚀 Quick API Reference

### Base URLs
```
Development:  http://localhost:8000/api/v1
Production:   https://api.facesocial.com/v1
WebSocket:    ws://localhost:8000/ws
```

### Authentication
All API endpoints require authentication except:
- `POST /auth/login`
- `POST /auth/register`
- `GET /health`
- `GET /metrics` (with API key)

### Common Headers
```http
Content-Type: application/json
Authorization: Bearer <jwt_token>
X-API-Version: v1
X-Request-ID: <uuid>
```

## 🔄 API Categories

### 1. Authentication & Authorization
```http
POST   /auth/login           # User login
POST   /auth/register        # User registration
POST   /auth/refresh         # Token refresh
POST   /auth/logout          # User logout
GET    /auth/profile         # User profile
PUT    /auth/profile         # Update profile
```

### 2. Face Analysis Services
```http
POST   /face/detect          # Face detection
POST   /face/recognize       # Face recognition
POST   /face/verify          # Face verification
POST   /face/analyze         # Complete face analysis
POST   /face/anti-spoof      # Anti-spoofing check
POST   /face/deepfake        # Deepfake detection
POST   /face/demographics    # Age/gender detection
```

### 3. Batch Processing
```http
POST   /batch/submit         # Submit batch job
GET    /batch/status/{id}    # Job status
GET    /batch/results/{id}   # Job results
DELETE /batch/cancel/{id}    # Cancel job
GET    /batch/list           # List user jobs
```

### 4. System Management
```http
GET    /system/health        # System health
GET    /system/status        # Detailed status
GET    /system/models        # Model information
POST   /system/models/load   # Load model
POST   /system/models/unload # Unload model
GET    /system/vram          # VRAM usage
POST   /system/optimize      # Optimize memory
```

### 5. Real-time WebSocket
```
/ws/face-analysis            # Real-time face analysis
/ws/system-status            # System monitoring
/ws/notifications            # User notifications
```

## 📊 Response Formats

### Success Response
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "meta": {
    "timestamp": "2025-06-01T10:00:00Z",
    "request_id": "req_123456",
    "processing_time": 150,
    "vram_usage": {
      "used": "2.1GB",
      "available": "3.9GB"
    }
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Face not detected in image",
    "details": {
      "field": "image",
      "validation": "face_required"
    }
  },
  "meta": {
    "timestamp": "2025-06-01T10:00:00Z",
    "request_id": "req_123456"
  }
}
```

## 🚨 Error Codes

### Authentication Errors (40x)
- `AUTH_REQUIRED` - Authentication required
- `AUTH_INVALID` - Invalid credentials
- `AUTH_EXPIRED` - Token expired
- `AUTH_FORBIDDEN` - Insufficient permissions

### Input Validation Errors (42x)
- `INVALID_INPUT` - Invalid request data
- `MISSING_FIELD` - Required field missing
- `INVALID_FORMAT` - Invalid data format
- `FILE_TOO_LARGE` - File size exceeded

### Processing Errors (50x)
- `PROCESSING_FAILED` - AI processing failed
- `MODEL_UNAVAILABLE` - Model not loaded
- `VRAM_EXHAUSTED` - Insufficient VRAM
- `QUEUE_FULL` - Processing queue full

## 📈 Rate Limiting

### Default Limits
- **Authenticated Users**: 1000 requests/hour
- **Face Analysis**: 100 requests/hour
- **Batch Processing**: 10 jobs/hour
- **WebSocket**: 50 connections/user

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1638360000
X-RateLimit-Retry-After: 3600
```

## 🔍 Request Tracing

All requests include tracing headers for debugging:
```http
X-Request-ID: req_abc123def456
X-Correlation-ID: corr_789xyz012
X-Processing-Time: 150ms
X-Model-Used: yolov10n-face
X-VRAM-Usage: 2.1GB/6GB
```

## 📝 API Versioning

### Version Strategy
- **URL Versioning**: `/api/v1/endpoint`
- **Header Versioning**: `X-API-Version: v1`
- **Deprecation Notice**: 6 months before removal

### Supported Versions
- **v1** (Current) - Full feature support
- **v0** (Deprecated) - Legacy endpoints

## 🛡️ Security Features

### Request Security
- **JWT Authentication** with RS256 signing
- **Rate Limiting** per user/IP
- **Input Validation** and sanitization
- **CORS** configuration
- **Request Size** limits

### Data Protection
- **Image Encryption** in transit
- **No Data Storage** of processed images
- **Audit Logging** of all requests
- **PII Anonymization** in logs

## 🔧 Development Tools

### API Testing
- **Postman Collection**: Available in `/docs/postman/`
- **OpenAPI Spec**: Available at `/docs/openapi.json`
- **Interactive Docs**: Available at `/docs/`

### SDK Support
- **Python SDK**: `pip install facesocial-sdk`
- **JavaScript SDK**: `npm install facesocial-sdk`
- **cURL Examples**: In each endpoint documentation

---

*For detailed endpoint documentation, refer to the individual files in this section.*
