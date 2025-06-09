# Authentication API

Comprehensive authentication and authorization system with JWT tokens, role-based access control, and security features.

## 🔐 Authentication Overview

### Security Features
- **JWT Tokens** with RS256 signing
- **Refresh Token** rotation
- **Role-Based Access Control** (RBAC)
- **Multi-Factor Authentication** (optional)
- **Session Management**
- **Audit Logging**

### Token Structure
```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_123",
    "email": "user@example.com",
    "role": "user",
    "permissions": ["face:analyze", "batch:submit"],
    "iat": 1638360000,
    "exp": 1638363600,
    "iss": "facesocial-api"
  }
}
```

## 🚀 Authentication Endpoints

### 1. User Login

**Endpoint:** `POST /auth/login`

**Description:** Authenticate user and receive JWT tokens

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "remember_me": false,
  "mfa_code": "123456"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJSUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJSUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "user": {
      "id": "user_123",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "user",
      "permissions": ["face:analyze", "batch:submit"],
      "last_login": "2025-06-01T10:00:00Z"
    }
  },
  "meta": {
    "timestamp": "2025-06-01T10:00:00Z",
    "request_id": "req_login_123"
  }
}
```

**Error Responses:**
```json
{
  "success": false,
  "error": {
    "code": "AUTH_INVALID",
    "message": "Invalid email or password"
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password"
  }'
```

### 2. User Registration

**Endpoint:** `POST /auth/register`

**Description:** Register new user account

**Request:**
```json
{
  "email": "newuser@example.com",
  "password": "secure_password",
  "confirm_password": "secure_password",
  "name": "Jane Doe",
  "company": "Acme Corp",
  "phone": "+1234567890",
  "terms_accepted": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "user_456",
      "email": "newuser@example.com",
      "name": "Jane Doe",
      "role": "user",
      "status": "pending_verification",
      "created_at": "2025-06-01T10:00:00Z"
    },
    "verification_required": true,
    "message": "Please check your email for verification link"
  }
}
```

**Validation Rules:**
- Email: Valid format, unique
- Password: Min 8 chars, complexity requirements
- Name: 2-50 characters
- Terms: Must be accepted

### 3. Token Refresh

**Endpoint:** `POST /auth/refresh`

**Description:** Refresh access token using refresh token

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJSUzI1NiIs..."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJSUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJSUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 3600
  }
}
```

### 4. User Logout

**Endpoint:** `POST /auth/logout`

**Description:** Logout user and invalidate tokens

**Headers:**
```http
Authorization: Bearer <access_token>
```

**Request:**
```json
{
  "all_devices": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Successfully logged out",
    "logout_timestamp": "2025-06-01T10:00:00Z"
  }
}
```

### 5. User Profile

**Endpoint:** `GET /auth/profile`

**Description:** Get current user profile

**Headers:**
```http
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "user_123",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "user",
      "permissions": ["face:analyze", "batch:submit"],
      "company": "Acme Corp",
      "phone": "+1234567890",
      "email_verified": true,
      "mfa_enabled": false,
      "last_login": "2025-06-01T10:00:00Z",
      "created_at": "2025-05-01T10:00:00Z",
      "usage_stats": {
        "api_calls_today": 45,
        "api_calls_month": 1250,
        "storage_used": "0MB"
      }
    }
  }
}
```

### 6. Update Profile

**Endpoint:** `PUT /auth/profile`

**Description:** Update user profile information

**Request:**
```json
{
  "name": "John Smith",
  "company": "New Company",
  "phone": "+1987654321",
  "preferences": {
    "notifications": true,
    "theme": "dark"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "user_123",
      "name": "John Smith",
      "company": "New Company",
      "updated_at": "2025-06-01T10:00:00Z"
    },
    "message": "Profile updated successfully"
  }
}
```

## 🔑 Password Management

### Change Password

**Endpoint:** `POST /auth/change-password`

**Request:**
```json
{
  "current_password": "old_password",
  "new_password": "new_secure_password",
  "confirm_password": "new_secure_password"
}
```

### Reset Password

**Endpoint:** `POST /auth/reset-password`

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Password reset email sent",
    "reset_token_expires": "2025-06-01T11:00:00Z"
  }
}
```

### Confirm Password Reset

**Endpoint:** `POST /auth/reset-password/confirm`

**Request:**
```json
{
  "reset_token": "reset_token_here",
  "new_password": "new_secure_password",
  "confirm_password": "new_secure_password"
}
```

## 🛡️ Security Features

### Multi-Factor Authentication

**Enable MFA:**
```http
POST /auth/mfa/enable
```

**Request:**
```json
{
  "method": "totp",
  "phone": "+1234567890"
}
```

**Verify MFA:**
```http
POST /auth/mfa/verify
```

**Request:**
```json
{
  "code": "123456",
  "backup_code": "abc123def456"
}
```

### Session Management

**List Active Sessions:**
```http
GET /auth/sessions
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sessions": [
      {
        "id": "session_123",
        "device": "Chrome on Windows",
        "ip_address": "192.168.1.100",
        "location": "New York, US",
        "last_activity": "2025-06-01T10:00:00Z",
        "current": true
      }
    ]
  }
}
```

**Revoke Session:**
```http
DELETE /auth/sessions/{session_id}
```

## 👥 Role-Based Access Control

### User Roles

| Role | Permissions | Description |
|------|-------------|-------------|
| `user` | Basic face analysis | Standard user |
| `premium` | Advanced features, batch processing | Premium subscriber |
| `admin` | System management | Administrator |
| `developer` | API access, webhooks | Developer account |

### Permission System

```json
{
  "permissions": [
    "face:analyze",      // Face analysis endpoints
    "face:batch",        // Batch processing
    "system:monitor",    // System monitoring
    "admin:users",       // User management
    "admin:system"       // System administration
  ]
}
```

### Check Permissions

**Endpoint:** `GET /auth/permissions`

**Response:**
```json
{
  "success": true,
  "data": {
    "permissions": ["face:analyze", "batch:submit"],
    "role": "user",
    "limits": {
      "api_calls_per_hour": 1000,
      "batch_jobs_per_hour": 10,
      "file_size_limit": "10MB"
    }
  }
}
```

## 📊 Usage Tracking

### API Usage Stats

**Endpoint:** `GET /auth/usage`

**Query Parameters:**
- `period`: `day`, `week`, `month`, `year`
- `start_date`: ISO date string
- `end_date`: ISO date string

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "month",
    "total_calls": 5000,
    "successful_calls": 4950,
    "failed_calls": 50,
    "by_endpoint": {
      "/face/analyze": 3000,
      "/face/detect": 1500,
      "/batch/submit": 500
    },
    "by_date": [
      {
        "date": "2025-06-01",
        "calls": 150
      }
    ]
  }
}
```

## 🚨 Error Handling

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `AUTH_REQUIRED` | Authentication required | 401 |
| `AUTH_INVALID` | Invalid credentials | 401 |
| `AUTH_EXPIRED` | Token expired | 401 |
| `AUTH_FORBIDDEN` | Insufficient permissions | 403 |
| `VALIDATION_ERROR` | Input validation failed | 422 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `USER_SUSPENDED` | Account suspended | 403 |
| `EMAIL_NOT_VERIFIED` | Email verification required | 403 |

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "AUTH_INVALID",
    "message": "Invalid email or password",
    "details": {
      "field": "password",
      "attempts_remaining": 2
    }
  },
  "meta": {
    "timestamp": "2025-06-01T10:00:00Z",
    "request_id": "req_auth_error_123"
  }
}
```

## 🔧 SDK Examples

### Python SDK

```python
from facesocial_sdk import FaceSocialClient

# Initialize client
client = FaceSocialClient()

# Login
response = client.auth.login(
    email="user@example.com",
    password="secure_password"
)

# Access token is automatically stored
token = response.data.access_token

# Get profile
profile = client.auth.get_profile()
print(f"Welcome, {profile.data.user.name}!")
```

### JavaScript SDK

```javascript
import { FaceSocialClient } from 'facesocial-sdk';

const client = new FaceSocialClient();

// Login
const loginResponse = await client.auth.login({
  email: 'user@example.com',
  password: 'secure_password'
});

// Token is automatically stored in client
console.log('Login successful:', loginResponse.data.user.name);

// Get profile
const profile = await client.auth.getProfile();
console.log('User permissions:', profile.data.user.permissions);
```

---

*Next: [02-face-analysis-api.md](02-face-analysis-api.md) - Face Analysis API Documentation*
