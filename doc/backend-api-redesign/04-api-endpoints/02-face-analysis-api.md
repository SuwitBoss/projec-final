# Face Analysis API

Comprehensive face analysis endpoints providing detection, recognition, verification, anti-spoofing, deepfake detection, and demographic analysis.

## 🎯 Face Analysis Overview

### Supported Analysis Types
- **Face Detection** - Locate faces in images/video
- **Face Recognition** - Identify known faces
- **Face Verification** - 1:1 face matching
- **Anti-Spoofing** - Liveness detection
- **Deepfake Detection** - Synthetic face detection
- **Demographics** - Age and gender estimation
- **Complete Analysis** - All analyses combined

### Input Formats
- **Image**: JPEG, PNG, WebP (max 10MB)
- **Video**: MP4, WebM, AVI (max 50MB, 30s)
- **Base64**: Encoded image data
- **URL**: Public image/video URLs

## 🚀 Core Analysis Endpoints

### 1. Face Detection

**Endpoint:** `POST /face/detect`

**Description:** Detect and locate faces in images or video frames

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "options": {
    "engine": "yolov10n",
    "confidence_threshold": 0.5,
    "max_faces": 10,
    "return_landmarks": true,
    "return_attributes": false
  }
}
```

**Alternative Request (File Upload):**
```http
POST /face/detect
Content-Type: multipart/form-data

image: <image_file>
options: {"engine": "yolov10n", "confidence_threshold": 0.5}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "faces_detected": 2,
    "faces": [
      {
        "face_id": "face_001",
        "bbox": {
          "x": 100,
          "y": 150,
          "width": 200,
          "height": 250
        },
        "confidence": 0.95,
        "landmarks": {
          "left_eye": {"x": 150, "y": 200},
          "right_eye": {"x": 220, "y": 200},
          "nose": {"x": 185, "y": 240},
          "mouth_left": {"x": 160, "y": 280},
          "mouth_right": {"x": 210, "y": 280}
        },
        "quality_score": 0.87
      }
    ],
    "image_info": {
      "width": 1024,
      "height": 768,
      "format": "JPEG"
    }
  },
  "meta": {
    "timestamp": "2025-06-01T10:00:00Z",
    "request_id": "req_detect_123",
    "processing_time": 85,
    "model_used": "yolov10n-face",
    "vram_usage": "1.2GB/6GB"
  }
}
```

**Engine Options:**
- `yolov10n` - Fast, lightweight (9MB VRAM)
- `yolov5s` - Balanced accuracy/speed (28MB VRAM)
- `mediapipe` - CPU-optimized, real-time
- `insightface` - High accuracy (GPU required)

### 2. Face Recognition

**Endpoint:** `POST /face/recognize`

**Description:** Identify faces against a known face database

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "options": {
    "model": "adaface_ir101",
    "confidence_threshold": 0.6,
    "max_results": 5,
    "database_id": "default",
    "return_embeddings": false
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "faces": [
      {
        "face_id": "face_001",
        "bbox": {
          "x": 100,
          "y": 150,
          "width": 200,
          "height": 250
        },
        "recognition_results": [
          {
            "person_id": "person_456",
            "person_name": "John Doe",
            "confidence": 0.92,
            "similarity_score": 0.89,
            "last_seen": "2025-05-30T14:30:00Z"
          }
        ],
        "unknown": false
      }
    ],
    "recognition_time": 120
  },
  "meta": {
    "model_used": "adaface_ir101",
    "database_size": 1500,
    "vram_usage": "2.1GB/6GB"
  }
}
```

**Model Options:**
- `adaface_ir101` - High accuracy (89MB VRAM)
- `arcface_r100` - Balanced (249MB VRAM)
- `facenet_vggface2` - Research grade (249MB VRAM)

### 3. Face Verification

**Endpoint:** `POST /face/verify`

**Description:** 1:1 face comparison between two images

**Request:**
```json
{
  "image1": "base64_encoded_image1",
  "image2": "base64_encoded_image2",
  "options": {
    "model": "adaface_ir101",
    "threshold": 0.6
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_same_person": true,
    "confidence": 0.94,
    "similarity_score": 0.91,
    "threshold_used": 0.6,
    "face1": {
      "detected": true,
      "quality_score": 0.87,
      "bbox": {"x": 100, "y": 150, "width": 200, "height": 250}
    },
    "face2": {
      "detected": true,
      "quality_score": 0.92,
      "bbox": {"x": 80, "y": 120, "width": 180, "height": 220}
    }
  }
}
```

### 4. Complete Face Analysis

**Endpoint:** `POST /face/analyze`

**Description:** Comprehensive face analysis including all available features

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "analyses": [
    "detection",
    "recognition",
    "anti_spoofing",
    "deepfake",
    "demographics"
  ],
  "options": {
    "detection_engine": "yolov10n",
    "recognition_model": "adaface_ir101",
    "database_id": "default",
    "return_detailed": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "faces": [
      {
        "face_id": "face_001",
        "bbox": {
          "x": 100,
          "y": 150,
          "width": 200,
          "height": 250
        },
        "detection": {
          "confidence": 0.95,
          "quality_score": 0.87,
          "landmarks": {
            "left_eye": {"x": 150, "y": 200},
            "right_eye": {"x": 220, "y": 200}
          }
        },
        "recognition": {
          "person_id": "person_456",
          "person_name": "John Doe",
          "confidence": 0.92,
          "unknown": false
        },
        "anti_spoofing": {
          "is_real": true,
          "confidence": 0.96,
          "spoof_type": null,
          "liveness_score": 0.94
        },
        "deepfake": {
          "is_deepfake": false,
          "confidence": 0.98,
          "manipulation_score": 0.02,
          "artifacts_detected": []
        },
        "demographics": {
          "age": {
            "value": 28,
            "range": "25-30",
            "confidence": 0.85
          },
          "gender": {
            "value": "male",
            "confidence": 0.92
          }
        }
      }
    ],
    "image_analysis": {
      "overall_quality": 0.89,
      "lighting_score": 0.92,
      "blur_score": 0.85,
      "resolution_adequate": true
    }
  },
  "meta": {
    "total_processing_time": 450,
    "models_used": {
      "detection": "yolov10n-face",
      "recognition": "adaface_ir101",
      "anti_spoofing": "minifasnet_v2",
      "deepfake": "efficientnet_b4",
      "demographics": "utkface"
    },
    "vram_usage": "3.2GB/6GB"
  }
}
```

## 🛡️ Anti-Spoofing Detection

### Anti-Spoofing Analysis

**Endpoint:** `POST /face/anti-spoof`

**Description:** Detect presentation attacks and verify face liveness

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "options": {
    "model": "minifasnet_v2",
    "threshold": 0.5,
    "motion_analysis": true,
    "depth_check": false
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "faces": [
      {
        "face_id": "face_001",
        "is_real": true,
        "confidence": 0.96,
        "liveness_score": 0.94,
        "spoof_detection": {
          "print_attack": 0.02,
          "replay_attack": 0.01,
          "mask_attack": 0.03
        },
        "motion_analysis": {
          "eye_blink": true,
          "head_movement": true,
          "micro_expressions": true
        }
      }
    ]
  },
  "meta": {
    "model_used": "minifasnet_v2",
    "processing_time": 95
  }
}
```

**Model Options:**
- `minifasnet_v2` - Lightweight, fast (1.9MB)
- `silent_face` - High accuracy (1.9MB)

## 🤖 Deepfake Detection

### Deepfake Analysis

**Endpoint:** `POST /face/deepfake`

**Description:** Detect AI-generated or manipulated faces

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "options": {
    "model": "efficientnet_b4",
    "threshold": 0.5,
    "artifact_analysis": true,
    "temporal_consistency": false
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "faces": [
      {
        "face_id": "face_001",
        "is_deepfake": false,
        "confidence": 0.98,
        "manipulation_score": 0.02,
        "artifacts": {
          "compression_artifacts": 0.01,
          "blending_artifacts": 0.02,
          "frequency_anomalies": 0.01
        },
        "quality_metrics": {
          "sharpness": 0.89,
          "contrast": 0.87,
          "naturalness": 0.94
        }
      }
    ]
  },
  "meta": {
    "model_used": "efficientnet_b4",
    "processing_time": 180
  }
}
```

**Model Options:**
- `efficientnet_b4` - Balanced accuracy/speed (44MB)
- `xceptionnet` - High accuracy (88MB)
- `mobilenet` - Fast inference (15MB)

## 👥 Demographics Analysis

### Age and Gender Detection

**Endpoint:** `POST /face/demographics`

**Description:** Estimate age and gender from facial features

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "options": {
    "model": "utkface",
    "return_probabilities": true,
    "age_grouping": "decade"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "faces": [
      {
        "face_id": "face_001",
        "age": {
          "value": 28,
          "range": "25-30",
          "confidence": 0.85,
          "probabilities": {
            "0-10": 0.01,
            "11-20": 0.05,
            "21-30": 0.78,
            "31-40": 0.14,
            "41-50": 0.02
          }
        },
        "gender": {
          "value": "male",
          "confidence": 0.92,
          "probabilities": {
            "male": 0.92,
            "female": 0.08
          }
        }
      }
    ]
  },
  "meta": {
    "model_used": "utkface",
    "processing_time": 65
  }
}
```

**Model Options:**
- `utkface` - Balanced (1.3MB)
- `fairface` - Bias-reduced (2.1MB)
- `agenet` - Age-focused (3.2MB)

## 📹 Video Analysis

### Video Face Analysis

**Endpoint:** `POST /face/analyze-video`

**Description:** Analyze faces across video frames

**Request:**
```json
{
  "video": "base64_encoded_video_data",
  "options": {
    "frame_sampling": "auto",
    "max_frames": 30,
    "tracking": true,
    "analyses": ["detection", "recognition"],
    "output_format": "timeline"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "video_info": {
      "duration": 15.5,
      "fps": 30,
      "frames_analyzed": 25,
      "resolution": "1920x1080"
    },
    "tracks": [
      {
        "track_id": "track_001",
        "person_id": "person_456",
        "person_name": "John Doe",
        "confidence": 0.89,
        "appearances": [
          {
            "frame": 10,
            "timestamp": 0.33,
            "bbox": {"x": 100, "y": 150, "width": 200, "height": 250}
          }
        ]
      }
    ],
    "timeline": [
      {
        "timestamp": 0.0,
        "faces_count": 1,
        "faces": ["track_001"]
      }
    ]
  },
  "meta": {
    "processing_time": 2500,
    "total_detections": 45
  }
}
```

## 📊 Batch Analysis

### Batch Face Analysis

**Endpoint:** `POST /face/batch-analyze`

**Description:** Analyze multiple images in a single request

**Request:**
```json
{
  "images": [
    {
      "id": "img_001",
      "data": "base64_encoded_image1"
    },
    {
      "id": "img_002",
      "data": "base64_encoded_image2"
    }
  ],
  "analyses": ["detection", "recognition"],
  "options": {
    "parallel_processing": true,
    "max_concurrent": 3
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "image_id": "img_001",
        "success": true,
        "faces": [
          {
            "face_id": "face_001",
            "detection": { /* detection results */ },
            "recognition": { /* recognition results */ }
          }
        ]
      }
    ],
    "summary": {
      "total_images": 2,
      "successful": 2,
      "failed": 0,
      "total_faces": 3
    }
  },
  "meta": {
    "batch_processing_time": 350,
    "parallel_processed": true
  }
}
```

## 🔄 Real-time Analysis

### Stream Analysis

**Endpoint:** `POST /face/stream-analyze`

**Description:** Analyze faces in real-time video streams

**Request:**
```json
{
  "stream_url": "rtmp://stream.example.com/live",
  "options": {
    "frame_rate": 5,
    "buffer_size": 10,
    "analyses": ["detection", "recognition"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "stream_id": "stream_123",
    "status": "active",
    "websocket_url": "ws://localhost:8000/ws/stream/stream_123"
  }
}
```

## 🚨 Error Handling

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `NO_FACE_DETECTED` | No faces found in image | 422 |
| `INVALID_IMAGE_FORMAT` | Unsupported image format | 422 |
| `IMAGE_TOO_LARGE` | Image exceeds size limit | 413 |
| `MODEL_UNAVAILABLE` | Requested model not loaded | 503 |
| `PROCESSING_FAILED` | Analysis processing error | 500 |
| `INSUFFICIENT_VRAM` | Not enough VRAM available | 503 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |

### Error Response Examples

```json
{
  "success": false,
  "error": {
    "code": "NO_FACE_DETECTED",
    "message": "No faces detected in the provided image",
    "details": {
      "min_face_size": "80x80",
      "confidence_threshold": 0.5,
      "suggestions": [
        "Ensure faces are clearly visible",
        "Check image quality and lighting",
        "Try lowering confidence threshold"
      ]
    }
  }
}
```

## 📈 Performance Optimization

### Request Optimization

**Batch Multiple Analyses:**
```json
{
  "image": "base64_data",
  "analyses": ["detection", "recognition", "demographics"],
  "options": {
    "optimize_memory": true,
    "reuse_detection": true
  }
}
```

**Model Preloading:**
```json
{
  "preload_models": ["yolov10n", "adaface_ir101"],
  "cache_duration": 3600
}
```

### Response Optimization

**Minimal Response:**
```json
{
  "options": {
    "return_minimal": true,
    "exclude_landmarks": true,
    "exclude_embeddings": true
  }
}
```

## 🔧 SDK Examples

### Python SDK

```python
from facesocial_sdk import FaceSocialClient
import base64

client = FaceSocialClient(api_key="your_api_key")

# Read and encode image
with open("face.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Complete face analysis
result = client.face.analyze(
    image=image_data,
    analyses=["detection", "recognition", "demographics"],
    options={
        "detection_engine": "yolov10n",
        "recognition_model": "adaface_ir101"
    }
)

for face in result.data.faces:
    print(f"Detected: {face.detection.confidence}")
    if not face.recognition.unknown:
        print(f"Recognized: {face.recognition.person_name}")
    print(f"Age: {face.demographics.age.value}")
    print(f"Gender: {face.demographics.gender.value}")
```

### JavaScript SDK

```javascript
import { FaceSocialClient } from 'facesocial-sdk';

const client = new FaceSocialClient({ apiKey: 'your_api_key' });

// Analyze uploaded file
const fileInput = document.getElementById('imageInput');
const file = fileInput.files[0];

const result = await client.face.analyzeFile(file, {
  analyses: ['detection', 'recognition'],
  options: {
    detection_engine: 'yolov10n',
    recognition_model: 'adaface_ir101'
  }
});

console.log('Faces detected:', result.data.faces.length);
result.data.faces.forEach(face => {
  console.log('Face confidence:', face.detection.confidence);
  if (!face.recognition.unknown) {
    console.log('Person:', face.recognition.person_name);
  }
});
```

---

*Next: [03-batch-processing-api.md](03-batch-processing-api.md) - Batch Processing API Documentation*
