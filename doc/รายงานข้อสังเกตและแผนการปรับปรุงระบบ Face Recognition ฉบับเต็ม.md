# 📋 รายงานข้อสังเกตและแผนการปรับปรุงระบบ Face Recognition ฉบับเต็ม

## 🚨 **ปัญหาร้ายแรงที่พบ**

### **1. ระบบไม่ใช้โมเดล AI จริง (Critical)**
```python
# ❌ ปัญหาใน face_recognition_service.py
async def extract_embedding(self, face_image) -> Optional[FaceEmbedding]:
    # Simulate embedding extraction
    await asyncio.sleep(0.05)
    
    # Generate fake embedding - สุ่มขึ้นมา!
    embedding_vector = np.random.randn(embedding_size).astype(np.float32)
    embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
```

**ผลกระทบ:**
- boss_04.jpg (ภาพเดียวกันที่ใช้เทรน) ได้แค่ 51.8% แทนที่จะเป็น >95%
- ระบบจดจำผิดแทบทุกภาพ
- ไม่สามารถแยกคนที่ไม่รู้จักได้

### **2. Model Loading เป็นการจำลอง (Critical)**
```python
# ❌ ปัญหาใน load_model()
async def load_model(self, model_type: ModelType) -> bool:
    # Simulate model loading - แค่จำลอง!
    await asyncio.sleep(0.1)
    self.current_model = f"{model_type.value.lower()}_model"  # แค่ string
```

**ไฟล์โมเดลที่มีแต่ไม่ได้ใช้:**
- `adaface_ir101.onnx` (89MB)
- `arcface_r100.onnx` (249MB)  
- `facenet_vggface2.onnx` (249MB)

---

## 🔍 **ปัญหารอง (High Priority)**

### **3. Detection Bounding Box ผิดปกติ**
```json
// ❌ bbox ครอบคลุมทั้งภาพ - ไม่แม่นยำ
"boss_01.jpg": {
  "bbox": {"x1": 0.0, "y1": 0.0, "x2": 2544.0, "y2": 3392.0}
}

// ✅ bbox ที่ถูกต้อง
"boss_03.jpg": {
  "bbox": {"x1": 528.08, "y1": 1377.01, "x2": 1876.87, "y2": 3054.22}
}
```

### **4. Quality Score คำนวณผิด**
```json
// ❌ Quality score เกิน 100
"quality_score": 789.0364074707031  // ควรเป็น 0-100

// ✅ Quality score ปกติ  
"quality_score": 95.15851211547852
```

### **5. ไม่มี Unknown Detection**
- ทุกใบหน้าถูกจำแนกเป็น "boss" หรือ "night" เสมอ
- ไม่มีการปฏิเสธใบหน้าที่ไม่รู้จัก
- Threshold ต่ำเกินไป (0.5) ควรเป็น 0.7-0.8

---

## 🛠️ **แผนการแก้ไขแบบละเอียด**

### **Phase 1: แก้ไขปัญหาร้ายแรง (Week 1)**

#### **1.1 เขียน Face Recognition Service ใหม่ทั้งหมด**
"""
Fixed Face Recognition Service - ใช้โมเดล ONNX จริง
แทนที่ fake embedding ด้วย real model inference
"""

import os
import time
import logging
import numpy as np
import cv2
import onnxruntime as ort
from typing import Optional, Dict, Any, List, Tuple
import asyncio
from dataclasses import dataclass

from .models import (
    FaceEmbedding, FaceMatch, FaceRecognitionResult, 
    FaceComparisonResult, ModelType
)

logger = logging.getLogger(__name__)


@dataclass
class RecognitionConfig:
    """Updated Configuration with Unknown Detection"""
    similarity_threshold: float = 0.75      # เพิ่มจาก 0.6
    unknown_threshold: float = 0.70         # เพิ่มใหม่
    max_faces: int = 10
    quality_threshold: float = 0.5          # เพิ่มจาก 0.3
    auto_model_selection: bool = True
    preferred_model: Optional[ModelType] = ModelType.FACENET
    enable_quality_assessment: bool = True
    enable_unknown_detection: bool = True   # เพิ่มใหม่


class FaceRecognitionService:
    """Real Face Recognition Service with ONNX Models"""
    
    def __init__(self, config: Optional[RecognitionConfig] = None, vram_manager=None):
        self.logger = logging.getLogger(__name__)
        self.config = config or RecognitionConfig()
        self.vram_manager = vram_manager
        
        # Real ONNX Models
        self.onnx_models: Dict[ModelType, ort.InferenceSession] = {}
        self.current_model_type: Optional[ModelType] = None
        
        # Model Paths
        self.model_paths = {
            ModelType.ADAFACE: "model/face-recognition/adaface_ir101.onnx",
            ModelType.ARCFACE: "model/face-recognition/arcface_r100.onnx", 
            ModelType.FACENET: "model/face-recognition/facenet_vggface2.onnx"
        }
        
        # Model Specifications
        self.model_specs = {
            ModelType.ADAFACE: {
                'input_size': (112, 112),
                'embedding_size': 512,
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]
            },
            ModelType.ARCFACE: {
                'input_size': (112, 112), 
                'embedding_size': 512,
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]
            },
            ModelType.FACENET: {
                'input_size': (160, 160),
                'embedding_size': 512,
                'mean': [127.5, 127.5, 127.5],
                'std': [128.0, 128.0, 128.0]
            }
        }
        
        # Face Database with Multiple Embeddings
        self.face_database: Dict[str, List[FaceEmbedding]] = {}
        
        # Performance Statistics
        self.stats = {
            'total_extractions': 0,
            'total_comparisons': 0,
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'unknown_detections': 0,
            'processing_times': []
        }
        
        self.logger.info("Real Face Recognition Service initialized")
    
    async def initialize(self) -> bool:
        """Initialize with real ONNX model loading"""
        try:
            # Select model
            if self.config.auto_model_selection:
                model_type = await self._select_best_model()
            else:
                model_type = self.config.preferred_model or ModelType.FACENET
            
            # Load real ONNX model
            success = await self.load_model(model_type)
            
            if success:
                self.logger.info(f"✅ Service initialized with REAL {model_type.value} model")
                return True
            else:
                self.logger.error("❌ Failed to initialize service")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing service: {e}")
            return False
    
    async def load_model(self, model_type: ModelType) -> bool:
        """Load real ONNX model (not simulation!)"""
        try:
            model_path = self.model_paths[model_type]
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            
            self.logger.info(f"🔄 Loading REAL ONNX model: {model_path}")
            
            # Configure ONNX Runtime providers
            providers = []
            if self.vram_manager:
                # Use GPU if available
                providers.extend(['CUDAExecutionProvider', 'CPUExecutionProvider'])
            else:
                providers.append('CPUExecutionProvider')
            
            # Session options for optimization
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Load real ONNX model
            self.onnx_models[model_type] = ort.InferenceSession(
                model_path,
                providers=providers,
                sess_options=session_options
            )
            
            self.current_model_type = model_type
            
            # Verify model inputs/outputs
            model = self.onnx_models[model_type]
            input_info = model.get_inputs()[0]
            output_info = model.get_outputs()[0]
            
            self.logger.info(f"✅ REAL model loaded successfully!")
            self.logger.info(f"   Input shape: {input_info.shape}")
            self.logger.info(f"   Output shape: {output_info.shape}")
            self.logger.info(f"   Providers: {model.get_providers()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load REAL model {model_type.value}: {e}")
            return False
    
    async def extract_embedding(self, face_image: np.ndarray) -> Optional[FaceEmbedding]:
        """Extract REAL embedding using ONNX model"""
        if self.current_model_type not in self.onnx_models:
            self.logger.error("❌ No model loaded!")
            return None
        
        start_time = time.time()
        
        try:
            # Get model and specifications
            model = self.onnx_models[self.current_model_type]
            specs = self.model_specs[self.current_model_type]
            
            # Preprocess image for specific model
            input_tensor = self._preprocess_image(face_image, self.current_model_type)
            
            # Run REAL model inference
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: input_tensor})
            
            # Extract embedding from output
            embedding_vector = outputs[0][0]  # Shape: (embedding_size,)
            
            # Normalize embedding
            embedding_vector = embedding_vector / (np.linalg.norm(embedding_vector) + 1e-8)
            
            # Calculate quality score
            quality_score = self._calculate_embedding_quality(embedding_vector)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_extractions'] += 1
            self.stats['processing_times'].append(processing_time)
            
            embedding = FaceEmbedding(
                vector=embedding_vector.astype(np.float32),
                model_type=self.current_model_type,
                quality_score=quality_score,
                extraction_time=processing_time
            )
            
            self.logger.debug(f"✅ REAL embedding extracted: {embedding_vector.shape}, quality: {quality_score:.3f}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ REAL embedding extraction failed: {e}")
            return None
    
    def _preprocess_image(self, face_image: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Preprocess image for specific model"""
        specs = self.model_specs[model_type]
        input_size = specs['input_size']
        
        try:
            # Resize to model input size
            if face_image.shape[:2] != input_size:
                face_resized = cv2.resize(face_image, input_size, interpolation=cv2.INTER_AREA)
            else:
                face_resized = face_image.copy()
            
            # Convert to float32
            face_float = face_resized.astype(np.float32)
            
            # Model-specific normalization
            if model_type == ModelType.FACENET:
                # FaceNet: [-1, 1]
                face_normalized = (face_float - 127.5) / 128.0
            else:
                # ArcFace/AdaFace: [0, 1] -> [-1, 1]
                face_normalized = (face_float / 255.0 - 0.5) / 0.5
            
            # Convert from HWC to CHW and add batch dimension
            input_tensor = np.transpose(face_normalized, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            return input_tensor
            
        except Exception as e:
            self.logger.error(f"❌ Image preprocessing failed: {e}")
            raise
    
    def _calculate_embedding_quality(self, embedding: np.ndarray) -> float:
        """Calculate embedding quality score (0-1)"""
        try:
            # Calculate embedding statistics
            magnitude = np.linalg.norm(embedding)
            variance = np.var(embedding)
            sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)
            
            # Quality metrics
            magnitude_score = min(magnitude, 1.0)  # Good embeddings have ~1.0 magnitude
            variance_score = min(variance * 10, 1.0)  # Higher variance = more informative
            sparsity_score = max(0, 1.0 - sparsity)  # Lower sparsity = better
            
            # Combined quality score
            quality = (magnitude_score * 0.4 + variance_score * 0.4 + sparsity_score * 0.2)
            
            return float(np.clip(quality, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default medium quality
    
    async def recognize_face(self, face_image: np.ndarray) -> FaceRecognitionResult:
        """Recognize face with Unknown Detection"""
        start_time = time.time()
        
        try:
            # Extract embedding
            embedding = await self.extract_embedding(face_image)
            if embedding is None:
                return self._create_failed_result(start_time, "Failed to extract embedding")
            
            # Search in database
            matches = []
            for person_id, stored_embeddings in self.face_database.items():
                best_similarity = 0.0
                best_embedding = None
                
                # Compare with all embeddings of this person
                for stored_embedding in stored_embeddings:
                    if stored_embedding.model_type != embedding.model_type:
                        continue
                    
                    similarity = self._cosine_similarity(
                        embedding.vector, stored_embedding.vector
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_embedding = stored_embedding
                
                # Create match if above threshold
                if best_similarity >= self.config.similarity_threshold:
                    match = FaceMatch(
                        person_id=person_id,
                        confidence=best_similarity,
                        embedding=best_embedding
                    )
                    matches.append(match)
            
            # Sort by confidence
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            # Unknown Detection Logic
            best_match = None
            if matches:
                candidate = matches[0]
                
                # Check against unknown threshold
                if (self.config.enable_unknown_detection and 
                    candidate.confidence < self.config.unknown_threshold):
                    # Mark as unknown
                    self.stats['unknown_detections'] += 1
                    self.logger.info(f"🔍 Unknown face detected (confidence: {candidate.confidence:.3f} < {self.config.unknown_threshold:.3f})")
                else:
                    best_match = candidate
                    self.stats['successful_recognitions'] += 1
                    self.logger.info(f"✅ Face recognized: {best_match.person_id} (confidence: {best_match.confidence:.3f})")
            else:
                self.stats['unknown_detections'] += 1
                self.logger.info("🔍 No matches found - Unknown face")
            
            # Update statistics
            self.stats['total_recognitions'] += 1
            processing_time = time.time() - start_time
            
            return FaceRecognitionResult(
                matches=matches,
                best_match=best_match,
                confidence=best_match.confidence if best_match else 0.0,
                processing_time=processing_time,
                model_used=self.current_model_type,
                face_embedding=embedding  # Include the extracted embedding
            )
            
        except Exception as e:
            self.logger.error(f"❌ Face recognition failed: {e}")
            return self._create_failed_result(start_time, str(e))
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Convert to 0-1 range (from -1 to 1)
            similarity = (similarity + 1.0) / 2.0
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    async def add_face_to_database(self, person_id: str, face_image: np.ndarray, 
                                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add face to database with multiple embeddings support"""
        try:
            embedding = await self.extract_embedding(face_image)
            if embedding is None:
                self.logger.error(f"❌ Failed to extract embedding for {person_id}")
                return False
            
            # Add metadata
            if metadata:
                embedding.metadata = metadata
            
            # Add to database (support multiple embeddings per person)
            if person_id not in self.face_database:
                self.face_database[person_id] = []
            
            self.face_database[person_id].append(embedding)
            
            total_embeddings = len(self.face_database[person_id])
            self.logger.info(f"✅ Added embedding for {person_id} (total: {total_embeddings})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add face to database: {e}")
            return False
    
    def _create_failed_result(self, start_time: float, error_message: str) -> FaceRecognitionResult:
        """Create failed recognition result"""
        return FaceRecognitionResult(
            matches=[],
            best_match=None,
            confidence=0.0,
            processing_time=time.time() - start_time,
            model_used=self.current_model_type,
            error=error_message
        )
    
    async def _select_best_model(self) -> ModelType:
        """Select best model based on available resources"""
        # For now, prefer FaceNet for balance of speed and accuracy
        # TODO: Integrate with VRAM manager for intelligent selection
        return ModelType.FACENET
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            processing_times = self.stats['processing_times']
            avg_time = np.mean(processing_times) if processing_times else 0.0
            
            recognition_rate = (
                self.stats['successful_recognitions'] / self.stats['total_recognitions']
                if self.stats['total_recognitions'] > 0 else 0.0
            )
            
            unknown_rate = (
                self.stats['unknown_detections'] / self.stats['total_recognitions']  
                if self.stats['total_recognitions'] > 0 else 0.0
            )
            
            return {
                'current_model': self.current_model_type.value if self.current_model_type else None,
                'model_loaded': self.current_model_type is not None,
                'database_size': len(self.face_database),
                'total_embeddings': sum(len(embs) for embs in self.face_database.values()),
                'performance': {
                    'total_extractions': self.stats['total_extractions'],
                    'total_recognitions': self.stats['total_recognitions'],
                    'successful_recognitions': self.stats['successful_recognitions'],
                    'unknown_detections': self.stats['unknown_detections'],
                    'recognition_rate': recognition_rate,
                    'unknown_detection_rate': unknown_rate,
                    'average_processing_time': avg_time
                },
                'thresholds': {
                    'similarity_threshold': self.config.similarity_threshold,
                    'unknown_threshold': self.config.unknown_threshold,
                    'quality_threshold': self.config.quality_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance stats: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear ONNX models
            for model in self.onnx_models.values():
                del model
            self.onnx_models.clear()
            
            # Clear database
            self.face_database.clear()
            
            # Reset state
            self.current_model_type = None
            
            self.logger.info("✅ Face Recognition Service cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")

#### **1.2 แก้ไข Detection Bounding Box Issues**
"""
Fixed Face Detection Utils
แก้ไขปัญหา Quality Score และ Bounding Box ที่ผิดปกติ
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def calculate_face_quality(detection_bbox, image_shape: Tuple[int, int]) -> float:
    """
    คำนวณคุณภาพของใบหน้า - แก้ไขให้คืนค่า 0-100
    
    Args:
        detection_bbox: BoundingBox object หรือ dict
        image_shape: (height, width) ของรูปภาพ
    
    Returns:
        คะแนนคุณภาพ 0-100 (ไม่เกิน 100)
    """
    try:
        # Extract coordinates
        if hasattr(detection_bbox, 'x1'):
            # BoundingBox object
            x1, y1, x2, y2 = detection_bbox.x1, detection_bbox.y1, detection_bbox.x2, detection_bbox.y2
            confidence = detection_bbox.confidence
        else:
            # Dict format
            x1 = detection_bbox.get('x1', 0)
            y1 = detection_bbox.get('y1', 0) 
            x2 = detection_bbox.get('x2', 0)
            y2 = detection_bbox.get('y2', 0)
            confidence = detection_bbox.get('confidence', 0)
        
        # Image dimensions
        img_height, img_width = image_shape[:2]
        
        # Face dimensions
        face_width = max(x2 - x1, 1)
        face_height = max(y2 - y1, 1)
        face_area = face_width * face_height
        
        # Image area
        image_area = img_width * img_height
        
        # ===== คะแนนคุณภาพย่อย =====
        
        # 1. Size Score (40%) - ขนาดใบหน้าเทียบกับรูปภาพ
        area_ratio = face_area / image_area
        if area_ratio > 0.1:  # ใบหน้าใหญ่มาก (>10% ของรูป)
            size_score = 100
        elif area_ratio > 0.05:  # ใบหน้าใหญ่ (5-10%)
            size_score = 90
        elif area_ratio > 0.02:  # ใบหน้าปานกลาง (2-5%)  
            size_score = 75
        elif area_ratio > 0.005:  # ใบหน้าเล็ก (0.5-2%)
            size_score = 50
        else:  # ใบหน้าเล็กมาก (<0.5%)
            size_score = 25
        
        # 2. Resolution Score (30%) - ความละเอียดของใบหน้า
        min_face_dimension = min(face_width, face_height)
        if min_face_dimension >= 200:  # ความละเอียดสูง
            resolution_score = 100
        elif min_face_dimension >= 100:  # ความละเอียดดี
            resolution_score = 85
        elif min_face_dimension >= 64:   # ความละเอียดพอใช้
            resolution_score = 70
        elif min_face_dimension >= 32:   # ความละเอียดต่ำ
            resolution_score = 50
        else:  # ความละเอียดต่ำมาก
            resolution_score = 20
        
        # 3. Confidence Score (20%) - ความมั่นใจของ detection
        confidence_score = min(confidence * 100, 100)  # Convert to 0-100
        
        # 4. Aspect Ratio Score (10%) - สัดส่วนใบหน้า
        aspect_ratio = face_width / face_height
        ideal_ratio = 0.8  # อัตราส่วนใบหน้าปกติ
        
        ratio_diff = abs(aspect_ratio - ideal_ratio)
        if ratio_diff < 0.1:  # สัดส่วนดีมาก
            aspect_score = 100
        elif ratio_diff < 0.2:  # สัดส่วนดี
            aspect_score = 80
        elif ratio_diff < 0.3:  # สัดส่วนพอใช้
            aspect_score = 60
        else:  # สัดส่วนผิดปกติ
            aspect_score = 30
        
        # ===== คำนวณคะแนนรวม =====
        final_score = (
            size_score * 0.40 +           # 40% น้ำหนักขนาด
            resolution_score * 0.30 +     # 30% น้ำหนักความละเอียด
            confidence_score * 0.20 +     # 20% น้ำหนัก confidence
            aspect_score * 0.10           # 10% น้ำหนักสัดส่วน
        )
        
        # ตรวจสอบให้แน่ใจว่าอยู่ในช่วง 0-100
        final_score = max(0.0, min(100.0, final_score))
        
        # Log debug info
        logger.debug(f"Quality calculation: area_ratio={area_ratio:.4f}, "
                    f"resolution={min_face_dimension}px, confidence={confidence:.3f}, "
                    f"aspect_ratio={aspect_ratio:.3f}, final_score={final_score:.1f}")
        
        return float(final_score)
        
    except Exception as e:
        logger.error(f"❌ Quality calculation failed: {e}")
        return 50.0  # Default medium quality


def validate_bounding_box(bbox, image_shape: Tuple[int, int]) -> bool:
    """
    ตรวจสอบความถูกต้องของ bounding box
    
    Args:
        bbox: BoundingBox object หรือ dict
        image_shape: (height, width) ของรูปภาพ
        
    Returns:
        True ถ้า bbox ถูกต้อง, False ถ้าผิดปกติ
    """
    try:
        # Extract coordinates
        if hasattr(bbox, 'x1'):
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        else:
            x1 = bbox.get('x1', 0)
            y1 = bbox.get('y1', 0)
            x2 = bbox.get('x2', 0) 
            y2 = bbox.get('y2', 0)
        
        img_height, img_width = image_shape[:2]
        
        # ตรวจสอบเงื่อนไขพื้นฐาน
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"❌ Invalid bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return False
        
        # ตรวจสอบขอบเขต
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            logger.warning(f"❌ Bbox out of bounds: ({x1},{y1})-({x2},{y2}) vs image ({img_width},{img_height})")
            return False
        
        # ตรวจสอบขนาดขั้นต่ำ
        width = x2 - x1
        height = y2 - y1
        
        if width < 16 or height < 16:
            logger.warning(f"❌ Bbox too small: {width}x{height}")
            return False
        
        # ตรวจสอบว่าครอบคลุมทั้งภาพหรือไม่ (ปัญหาที่พบ)
        area_ratio = (width * height) / (img_width * img_height)
        if area_ratio > 0.9:  # ถ้าครอบคลุม >90% ของภาพ
            logger.warning(f"❌ Bbox covers entire image: {area_ratio:.3f}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bbox validation failed: {e}")
        return False


def filter_detection_results(faces: list, image_shape: Tuple[int, int], 
                           min_quality: float = 50.0) -> list:
    """
    กรองผลลัพธ์ detection โดยใช้เกณฑ์คุณภาพ
    
    Args:
        faces: รายการ FaceDetection objects
        image_shape: ขนาดรูปภาพ
        min_quality: คะแนนคุณภาพขั้นต่ำ (0-100)
        
    Returns:
        รายการใบหน้าที่ผ่านการกรอง
    """
    filtered_faces = []
    
    for face in faces:
        try:
            # ตรวจสอบ bounding box
            if not validate_bounding_box(face.bbox, image_shape):
                logger.debug(f"🚫 Face filtered: invalid bbox")
                continue
            
            # คำนวณคุณภาพใหม่ถ้าจำเป็น
            if face.quality_score is None or face.quality_score > 100:
                face.quality_score = calculate_face_quality(face.bbox, image_shape)
            
            # กรองตามคุณภาพ
            if face.quality_score >= min_quality:
                filtered_faces.append(face)
                logger.debug(f"✅ Face accepted: quality={face.quality_score:.1f}")
            else:
                logger.debug(f"🚫 Face filtered: quality={face.quality_score:.1f} < {min_quality}")
                
        except Exception as e:
            logger.error(f"❌ Error filtering face: {e}")
            continue
    
    logger.info(f"🔍 Face filtering: {len(faces)} -> {len(filtered_faces)} faces")
    
    return filtered_faces


def improve_detection_accuracy(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ปรับปรุงการตั้งค่าสำหรับความแม่นยำที่ดีขึ้น
    
    Args:
        config: การตั้งค่าปัจจุบัน
        
    Returns:
        การตั้งค่าที่ปรับปรุงแล้ว
    """
    improved_config = config.copy()
    
    # ปรับ detection parameters
    improved_config.update({
        # เพิ่ม confidence threshold เพื่อลด false positives
        'conf_threshold': max(0.4, config.get('conf_threshold', 0.15)),
        
        # เพิ่ม IoU threshold เพื่อลด duplicate detections  
        'iou_threshold': max(0.6, config.get('iou_threshold', 0.4)),
        
        # เพิ่มขนาดขั้นต่ำของใบหน้า
        'min_face_size': max(32, config.get('min_face_size', 16)),
        
        # เพิ่มเกณฑ์คุณภาพ
        'min_quality_threshold': max(60, config.get('min_quality_threshold', 50)),
        
        # ปรับการตั้งค่าการตัดสินใจโมเดล
        'max_usable_faces_yolov9': min(6, config.get('max_usable_faces_yolov9', 8)),
        'min_agreement_ratio': max(0.8, config.get('min_agreement_ratio', 0.7)),
    })
    
    logger.info("🔧 Detection config improved:")
    for key, value in improved_config.items():
        if key in config and config[key] != value:
            logger.info(f"   {key}: {config[key]} -> {value}")
    
    return improved_config


@dataclass
class QualityAnalysisResult:
    """ผลลัพธ์การวิเคราะห์คุณภาพ"""
    total_faces: int
    valid_faces: int
    high_quality_faces: int  # >80
    medium_quality_faces: int  # 60-80
    low_quality_faces: int  # <60
    average_quality: float
    quality_distribution: Dict[str, int]
    
    def __post_init__(self):
        self.valid_ratio = self.valid_faces / self.total_faces if self.total_faces > 0 else 0
        self.high_quality_ratio = self.high_quality_faces / self.total_faces if self.total_faces > 0 else 0


def analyze_detection_quality(faces: list, image_shape: Tuple[int, int]) -> QualityAnalysisResult:
    """
    วิเคราะห์คุณภาพการ detection โดยรวม
    
    Args:
        faces: รายการ FaceDetection objects
        image_shape: ขนาดรูปภาพ
        
    Returns:
        ผลลัพธ์การวิเคราะห์คุณภาพ
    """
    if not faces:
        return QualityAnalysisResult(
            total_faces=0, valid_faces=0, high_quality_faces=0,
            medium_quality_faces=0, low_quality_faces=0,
            average_quality=0.0, quality_distribution={}
        )
    
    valid_faces = 0
    high_quality = 0
    medium_quality = 0
    low_quality = 0
    quality_scores = []
    
    for face in faces:
        # คำนวณคุณภาพถ้ายังไม่มีหรือผิดปกติ
        if face.quality_score is None or face.quality_score > 100:
            face.quality_score = calculate_face_quality(face.bbox, image_shape)
        
        # ตรวจสอบความถูกต้อง
        if validate_bounding_box(face.bbox, image_shape):
            valid_faces += 1
            quality_scores.append(face.quality_score)
            
            # จำแนกระดับคุณภาพ
            if face.quality_score >= 80:
                high_quality += 1
            elif face.quality_score >= 60:
                medium_quality += 1
            else:
                low_quality += 1
    
    # คำนวณสถิติ
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
    
    # การกระจายของคุณภาพ
    quality_distribution = {
        'excellent': sum(1 for q in quality_scores if q >= 90),
        'good': sum(1 for q in quality_scores if 80 <= q < 90),
        'fair': sum(1 for q in quality_scores if 60 <= q < 80),
        'poor': sum(1 for q in quality_scores if q < 60)
    }
    
    return QualityAnalysisResult(
        total_faces=len(faces),
        valid_faces=valid_faces,
        high_quality_faces=high_quality,
        medium_quality_faces=medium_quality,
        low_quality_faces=low_quality,
        average_quality=avg_quality,
        quality_distribution=quality_distribution
    )


def create_quality_report(analysis: QualityAnalysisResult) -> str:
    """สร้างรายงานคุณภาพการ detection"""
    report = f"""
📊 Face Detection Quality Report
=====================================
Total Faces Detected: {analysis.total_faces}
Valid Faces: {analysis.valid_faces} ({analysis.valid_ratio:.1%})
Average Quality: {analysis.average_quality:.1f}/100

Quality Distribution:
├─ Excellent (90-100): {analysis.quality_distribution['excellent']} faces
├─ Good (80-89): {analysis.quality_distribution['good']} faces  
├─ Fair (60-79): {analysis.quality_distribution['fair']} faces
└─ Poor (<60): {analysis.quality_distribution['poor']} faces

High Quality Ratio: {analysis.high_quality_ratio:.1%}
"""
    return report.strip()
#### **1.3 ปรับปรุง Face Detection Service**

"""
Improved Face Detection Service
แก้ไขปัญหา bounding box และความแม่นยำ
"""

import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Union

from .yolo_models import YOLOv9ONNXDetector, YOLOv11Detector
from .utils import (
    BoundingBox, FaceDetection, DetectionResult, 
    calculate_face_quality, validate_bounding_box,
    filter_detection_results, improve_detection_accuracy,
    analyze_detection_quality, create_quality_report
)

logger = logging.getLogger(__name__)


class ImprovedFaceDetectionService:
    """
    ปรับปรุง Face Detection Service ด้วยความแม่นยำที่ดีขึ้น
    """
    
    def __init__(self, vram_manager, config: Dict[str, Any]):
        """
        Initialize with improved configuration
        """
        self.vram_manager = vram_manager
        
        # ปรับปรุงการตั้งค่าอัตโนมัติ
        self.config = improve_detection_accuracy(config)
        
        self.models: Dict[str, Union[YOLOv9ONNXDetector, YOLOv11Detector]] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
        # เกณฑ์การตัดสินใจที่ปรับปรุงแล้ว
        self.decision_criteria = {
            'max_usable_faces_yolov9': self.config.get('max_usable_faces_yolov9', 6),  # ลดจาก 8
            'min_agreement_ratio': self.config.get('min_agreement_ratio', 0.8),      # เพิ่มจาก 0.7
            'min_quality_threshold': self.config.get('min_quality_threshold', 60),   # เพิ่มจาก 50
            'iou_threshold': self.config.get('iou_threshold', 0.6)                   # เพิ่มจาก 0.5
        }
        
        # พารามิเตอร์การตรวจจับที่ปรับปรุงแล้ว
        self.detection_params = {
            'conf_threshold': self.config.get('conf_threshold', 0.4),  # เพิ่มจาก 0.15
            'iou_threshold': self.config.get('iou_threshold', 0.6),    # เพิ่มจาก 0.4
            'img_size': self.config.get('img_size', 640)
        }
        
        # เพิ่มสถิติการทำงาน
        self.performance_stats = {
            'total_detections': 0,
            'valid_detections': 0,
            'filtered_detections': 0,
            'model_usage': {'yolov9c': 0, 'yolov9e': 0, 'yolov11m': 0},
            'average_quality': 0.0,
            'processing_times': []
        }
        
        self.models_loaded = False
        
        logger.info("🔧 Improved Face Detection Service initialized")
        logger.info(f"📊 Enhanced detection params: {self.detection_params}")
    
    async def initialize(self) -> bool:
        """โหลดโมเดลตรวจจับใบหน้าทั้งหมด"""
        try:
            logger.info("🚀 Loading improved face detection models...")
            
            # โหลดโมเดลเหมือนเดิม แต่ใช้ parameters ที่ปรับปรุงแล้ว
            success = await self._load_all_models()
            
            if success:
                self.models_loaded = True
                logger.info("✅ Improved face detection models loaded successfully")
                return True
            else:
                logger.error("❌ Failed to load models")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    async def detect_faces(self, 
                         image_input: Union[str, np.ndarray],
                         model_name: Optional[str] = None,
                         conf_threshold: Optional[float] = None,
                         iou_threshold: Optional[float] = None,
                         enhanced_mode: bool = True) -> DetectionResult:
        """
        ตรวจจับใบหน้าด้วยความแม่นยำที่ปรับปรุงแล้ว
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call initialize() first.")
        
        # ใช้ parameters ที่ปรับปรุงแล้ว
        conf_threshold = conf_threshold or self.detection_params['conf_threshold']
        iou_threshold = iou_threshold or self.detection_params['iou_threshold']
        
        start_time = time.time()
        
        try:
            # โหลดรูปภาพ
            image = self._load_image(image_input)
            
            # ตรวจจับใบหน้า
            if model_name in ['yolov9c', 'yolov9e', 'yolov11m']:
                detections = self._detect_with_specific_model(
                    image, model_name, conf_threshold, iou_threshold
                )
                model_used = model_name
            else:
                # ใช้ระบบอัจฉริยะที่ปรับปรุงแล้ว
                detections, model_used = await self._enhanced_intelligent_detect(
                    image, conf_threshold, iou_threshold
                )
            
            # กรองและปรับปรุงผลลัพธ์
            detections = self._post_process_detections(detections, image.shape)
            
            total_time = time.time() - start_time
            
            # อัปเดตสถิติ
            self._update_performance_stats(detections, model_used, total_time)
            
            # สร้างผลลัพธ์
            result = DetectionResult(
                faces=detections,
                image_shape=(image.shape[0], image.shape[1], image.shape[2] if len(image.shape) > 2 else 3),
                total_processing_time=total_time,
                model_used=model_used,
                fallback_used=False
            )
            
            # เพิ่มข้อมูลคุณภาพ
            quality_analysis = analyze_detection_quality(detections, image.shape)
            result.quality_analysis = quality_analysis
            
            logger.info(f"✅ Detection complete: {len(detections)} faces, "
                       f"avg quality: {quality_analysis.average_quality:.1f}, "
                       f"model: {model_used}, time: {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            raise
    
    def _detect_with_specific_model(self, 
                                   image: np.ndarray, 
                                   model_name: str,
                                   conf_threshold: float,
                                   iou_threshold: float) -> List[FaceDetection]:
        """ตรวจจับใบหน้าด้วยโมเดลที่ระบุ - พร้อมการปรับปรุง"""
        start_time = time.time()
        
        try:
            # ตรวจจับด้วยโมเดลที่เลือก
            detections_raw = self.models[model_name].detect(
                image, conf_threshold, iou_threshold
            )
            
            inference_time = time.time() - start_time
            
            # แปลงผลลัพธ์พร้อมตรวจสอบคุณภาพ
            face_detections = []
            for det in detections_raw:
                bbox = BoundingBox.from_array(det)
                
                # ตรวจสอบ bbox ให้ถูกต้อง
                if not validate_bounding_box(bbox, image.shape):
                    logger.debug(f"🚫 Invalid bbox filtered: {bbox.to_array()}")
                    continue
                
                # คำนวณคุณภาพอย่างถูกต้อง
                quality_score = calculate_face_quality(bbox, image.shape)
                
                face = FaceDetection(
                    bbox=bbox,
                    quality_score=quality_score,
                    model_used=model_name,
                    processing_time=inference_time
                )
                face_detections.append(face)
            
            # บันทึกสถิติ
            self.model_stats[model_name] = {
                'last_inference_time': inference_time,
                'face_count': len(face_detections),
                'valid_faces': len(face_detections),
                'avg_quality': np.mean([f.quality_score for f in face_detections]) if face_detections else 0.0
            }
            
            logger.debug(f"🔍 {model_name}: {len(detections_raw)} raw -> {len(face_detections)} valid faces")
            
            return face_detections
            
        except Exception as e:
            logger.error(f"❌ Detection with {model_name} failed: {e}")
            return []
    
    async def _enhanced_intelligent_detect(self,
                                         image: np.ndarray,
                                         conf_threshold: float,
                                         iou_threshold: float) -> tuple[List[FaceDetection], str]:
        """
        ระบบตรวจจับอัจฉริยะที่ปรับปรุงแล้ว
        """
        logger.debug("🧠 Using enhanced intelligent detection...")
        
        # ขั้นตอน 1: ทดสอบด้วย YOLOv9c (เร็วที่สุด)
        yolov9c_detections = self._detect_with_specific_model(
            image, 'yolov9c', conf_threshold, iou_threshold
        )
        
        # ถ้าไม่พบใบหน้าเลย ลองใช้ YOLOv11m
        if not yolov9c_detections:
            logger.debug("🔄 YOLOv9c found no faces, trying YOLOv11m...")
            yolov11m_detections = self._detect_with_specific_model(
                image, 'yolov11m', conf_threshold, iou_threshold
            )
            return yolov11m_detections, 'yolov11m'
        
        # ถ้าพบใบหน้าน้อย ให้ใช้ YOLOv9c เลย
        max_faces = self.decision_criteria['max_usable_faces_yolov9']
        if len(yolov9c_detections) <= max_faces:
            logger.debug(f"✅ YOLOv9c sufficient: {len(yolov9c_detections)} faces ≤ {max_faces}")
            return yolov9c_detections, 'yolov9c'
        
        # ถ้าพบเยอะ ให้ทดสอบด้วย YOLOv9e
        logger.debug(f"🔄 YOLOv9c found many faces ({len(yolov9c_detections)}), testing YOLOv9e...")
        yolov9e_detections = self._detect_with_specific_model(
            image, 'yolov9e', conf_threshold, iou_threshold
        )
        
        # เปรียบเทียบผลลัพธ์ที่ปรับปรุงแล้ว
        agreement = self._calculate_improved_agreement(
            yolov9c_detections, yolov9e_detections
        )
        
        min_agreement = self.decision_criteria['min_agreement_ratio']
        if agreement >= min_agreement:
            logger.debug(f"✅ YOLOv9 models agree: {agreement:.1%} ≥ {min_agreement:.1%}")
            # เลือกโมเดลที่ให้คุณภาพดีกว่า
            return self._select_better_detection(yolov9c_detections, yolov9e_detections)
        
        # ถ้าไม่เห็นด้วยกัน ใช้ YOLOv11m
        logger.debug(f"🔄 YOLOv9 models disagree ({agreement:.1%}), using YOLOv11m...")
        yolov11m_detections = self._detect_with_specific_model(
            image, 'yolov11m', conf_threshold, iou_threshold
        )
        
        return yolov11m_detections, 'yolov11m'
    
    def _calculate_improved_agreement(self, 
                                    detections1: List[FaceDetection], 
                                    detections2: List[FaceDetection]) -> float:
        """คำนวณความเห็นด้วยที่ปรับปรุงแล้ว"""
        if not detections1 or not detections2:
            return 0.0
        
        # ใช้เฉพาะ detections ที่มีคุณภาพดี
        quality_threshold = self.decision_criteria['min_quality_threshold']
        
        good_detections1 = [d for d in detections1 if d.quality_score >= quality_threshold]
        good_detections2 = [d for d in detections2 if d.quality_score >= quality_threshold]
        
        if not good_detections1 or not good_detections2:
            return 0.0
        
        # คำนวณ overlap ระหว่าง high-quality faces
        total_faces = max(len(good_detections1), len(good_detections2))
        iou_threshold = self.decision_criteria['iou_threshold']
        
        matched_count = 0
        for face1 in good_detections1:
            for face2 in good_detections2:
                iou = self._calculate_bbox_iou(face1.bbox, face2.bbox)
                if iou >= iou_threshold:
                    matched_count += 1
                    break  # หา match แล้วข้ามไปหาคู่ต่อไป
        
        agreement = matched_count / total_faces
        logger.debug(f"🤝 Agreement: {matched_count}/{total_faces} = {agreement:.1%}")
        
        return agreement
    
    def _select_better_detection(self, 
                               detections1: List[FaceDetection], 
                               detections2: List[FaceDetection]) -> tuple[List[FaceDetection], str]:
        """เลือก detection ที่ดีกว่า"""
        # คำนวณคุณภาพเฉลี่ย
        avg_quality1 = np.mean([d.quality_score for d in detections1]) if detections1 else 0
        avg_quality2 = np.mean([d.quality_score for d in detections2]) if detections2 else 0
        
        # นับจำนวนใบหน้าคุณภาพสูง
        high_quality1 = sum(1 for d in detections1 if d.quality_score >= 80)
        high_quality2 = sum(1 for d in detections2 if d.quality_score >= 80)
        
        # ตัดสินใจตามคุณภาพ
        if high_quality2 > high_quality1:
            logger.debug(f"✅ YOLOv9e selected: {high_quality2} vs {high_quality1} high-quality faces")
            return detections2, 'yolov9e'
        elif high_quality1 > high_quality2:
            logger.debug(f"✅ YOLOv9c selected: {high_quality1} vs {high_quality2} high-quality faces")
            return detections1, 'yolov9c'
        else:
            # ถ้าเท่ากัน เลือกตามคุณภาพเฉลี่ย
            if avg_quality2 > avg_quality1:
                logger.debug(f"✅ YOLOv9e selected: avg quality {avg_quality2:.1f} vs {avg_quality1:.1f}")
                return detections2, 'yolov9e'
            else:
                logger.debug(f"✅ YOLOv9c selected: avg quality {avg_quality1:.1f} vs {avg_quality2:.1f}")
                return detections1, 'yolov9c'
    
    def _post_process_detections(self, 
                               detections: List[FaceDetection], 
                               image_shape: tuple) -> List[FaceDetection]:
        """ประมวลผลหลังการตรวจจับ"""
        if not detections:
            return detections
        
        # กรองตามคุณภาพ
        min_quality = self.decision_criteria['min_quality_threshold']
        filtered_detections = filter_detection_results(detections, image_shape, min_quality)
        
        # เรียงลำดับตามคุณภาพ
        filtered_detections.sort(key=lambda x: x.quality_score, reverse=True)
        
        # จำกัดจำนวนสูงสุด
        max_faces = self.config.get('max_faces', 50)
        if len(filtered_detections) > max_faces:
            logger.info(f"🔢 Limiting faces: {len(filtered_detections)} -> {max_faces}")
            filtered_detections = filtered_detections[:max_faces]
        
        return filtered_detections
    
    def _calculate_bbox_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """คำนวณ IoU ระหว่าง bounding boxes"""
        try:
            # หาพื้นที่ทับซ้อน
            x_left = max(bbox1.x1, bbox2.x1)
            y_top = max(bbox1.y1, bbox2.y1)
            x_right = min(bbox1.x2, bbox2.x2)
            y_bottom = min(bbox1.y2, bbox2.y2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # หาพื้นที่รวม
            area1 = bbox1.area
            area2 = bbox2.area
            union_area = area1 + area2 - intersection_area
            
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
            
        except Exception:
            return 0.0
    
    def _update_performance_stats(self, 
                                detections: List[FaceDetection], 
                                model_used: str, 
                                processing_time: float):
        """อัปเดตสถิติประสิทธิภาพ"""
        self.performance_stats['total_detections'] += len(detections)
        self.performance_stats['model_usage'][model_used] += 1
        self.performance_stats['processing_times'].append(processing_time)
        
        if detections:
            qualities = [d.quality_score for d in detections]
            current_avg = np.mean(qualities)
            
            # อัปเดตคุณภาพเฉลี่ย (rolling average)
            total_calls = sum(self.performance_stats['model_usage'].values())
            if total_calls == 1:
                self.performance_stats['average_quality'] = current_avg
            else:
                self.performance_stats['average_quality'] = (
                    (self.performance_stats['average_quality'] * (total_calls - 1) + current_avg) / total_calls
                )
    
    def get_enhanced_service_info(self) -> Dict[str, Any]:
        """ดูข้อมูลบริการที่ปรับปรุงแล้ว"""
        try:
            base_info = self.get_service_info()  # เรียกฟังก์ชันเดิม
            
            # เพิ่มข้อมูลการปรับปรุง
            enhanced_info = {
                **base_info,
                'enhanced_features': {
                    'improved_quality_filtering': True,
                    'bbox_validation': True,
                    'intelligent_model_selection': True,
                    'performance_monitoring': True
                },
                'performance_stats': self.performance_stats,
                'quality_thresholds': {
                    'min_quality': self.decision_criteria['min_quality_threshold'],
                    'agreement_ratio': self.decision_criteria['min_agreement_ratio'],
                    'iou_threshold': self.decision_criteria['iou_threshold']
                }
            }
            
            return enhanced_info
            
        except Exception as e:
            logger.error(f"❌ Error getting enhanced service info: {e}")
            return {'error': str(e)}
    
    def _load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """โหลดรูปภาพพร้อมตรวจสอบ"""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Cannot read image: {image_input}")
        else:
            image = image_input
            
        # ตรวจสอบขนาดรูปภาพ
        if image.shape[0] < 32 or image.shape[1] < 32:
            raise ValueError(f"Image too small: {image.shape}")
            
        return image
    
    async def _load_all_models(self) -> bool:
        """โหลดโมเดลทั้งหมดพร้อมการจัดการ error"""
        try:
            # โหลดโมเดลแต่ละตัว (implementation ขึ้นกับโค้ดเดิม)
            # ... existing model loading code ...
            return True
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
### **Phase 2: แผนการทดสอบและ Validation (Week 2)**---
# 🧪 แผนการทดสอบระบบ Face Recognition แบบครบถ้วน

## **เป้าหมายการทดสอบ**

### **1. ทดสอบการแก้ไขปัญหาหลัก**
- ✅ boss_04.jpg ต้องได้ confidence >95% (จากเดิม 51.8%)
- ✅ Unknown faces ต้องถูกจำแนกเป็น "unknown"
- ✅ Bounding boxes ต้องแม่นยำ (ไม่ครอบคลุมทั้งภาพ)
- ✅ Quality scores ต้องอยู่ในช่วง 0-100

### **2. ทดสอบประสิทธิภาพโดยรวม**
- Recognition accuracy >85%
- Processing time <2 วินาที/ภาพ
- False positive rate <10%
- Unknown detection rate เหมาะสม

---

## **ขั้นตอนการทดสอบ**

### **Step 1: Unit Testing - แต่ละส่วนย่อย**

#### **1.1 ทดสอบ Real ONNX Models**
```python
async def test_real_embedding_extraction():
    """ทดสอบการสกัด embedding จากโมเดลจริง"""
    service = FaceRecognitionService()
    await service.initialize()
    
    # โหลดภาพทดสอบ
    boss_image = cv2.imread("test_images/boss_04.jpg")
    
    # สกัด embedding 2 ครั้ง
    embedding1 = await service.extract_embedding(boss_image)
    embedding2 = await service.extract_embedding(boss_image)
    
    # ตรวจสอบความสอดคล้อง
    similarity = service._cosine_similarity(embedding1.vector, embedding2.vector)
    
    # ภาพเดียวกันต้องได้ similarity >95%
    assert similarity > 0.95, f"Same image similarity too low: {similarity:.3f}"
    print(f"✅ Same image similarity: {similarity:.3f}")
```

#### **1.2 ทดสอบ Quality Score Calculation**
```python
def test_quality_score_range():
    """ทดสอบช่วงคะแนนคุณภาพ"""
    # Test cases with known results
    test_cases = [
        {
            'bbox': {'x1': 100, 'y1': 100, 'x2': 300, 'y2': 300, 'confidence': 0.9},
            'image_shape': (640, 480),
            'expected_range': (70, 100)
        },
        {
            'bbox': {'x1': 0, 'y1': 0, 'x2': 640, 'y2': 480, 'confidence': 0.3},
            'image_shape': (640, 480),
            'expected_range': (0, 30)  # Full image bbox should be poor
        }
    ]
    
    for case in test_cases:
        quality = calculate_face_quality(case['bbox'], case['image_shape'])
        min_expected, max_expected = case['expected_range']
        
        assert 0 <= quality <= 100, f"Quality out of range: {quality}"
        assert min_expected <= quality <= max_expected, f"Quality {quality} not in expected range {case['expected_range']}"
        print(f"✅ Quality test passed: {quality:.1f}")
```

#### **1.3 ทดสอบ Unknown Detection**
```python
async def test_unknown_detection():
    """ทดสอบการจำแนกใบหน้าที่ไม่รู้จัก"""
    service = FaceRecognitionService()
    await service.initialize()
    
    # เพิ่มใบหน้าที่รู้จัก
    known_face = cv2.imread("test_images/boss_01.jpg")
    await service.add_face_to_database("boss", known_face)
    
    # ทดสอบกับใบหน้าที่ไม่รู้จัก
    unknown_face = cv2.imread("test_images/unknown_person.jpg")
    result = await service.recognize_face(unknown_face)
    
    # ต้องไม่มี best_match หรือ confidence ต่ำมาก
    assert result.best_match is None or result.confidence < 0.7, f"Unknown face wrongly recognized: {result.confidence}"
    print(f"✅ Unknown detection works: confidence={result.confidence:.3f}")
```

### **Step 2: Integration Testing - ระบบรวม**

#### **2.1 ทดสอบ End-to-End Pipeline**
```python
async def test_full_pipeline():
    """ทดสอบ pipeline ครบวงจร Detection + Recognition"""
    detection_service = ImprovedFaceDetectionService(vram_manager, detection_config)
    recognition_service = FaceRecognitionService(recognition_config)
    
    await detection_service.initialize()
    await recognition_service.initialize()
    
    # สร้าง gallery
    await build_multi_embedding_gallery(recognition_service, detection_service, "test_images")
    
    # ทดสอบภาพต่างๆ
    test_images = [
        ("test_images/boss_04.jpg", "boss", 0.95),  # ควรได้ >95%
        ("test_images/night_02.jpg", "night", 0.95),
        ("test_images/unknown_person.jpg", "unknown", 0.0)
    ]
    
    for image_path, expected_identity, min_confidence in test_images:
        image = cv2.imread(image_path)
        
        # Detection
        detection_result = await detection_service.detect_faces(image)
        assert len(detection_result.faces) > 0, f"No faces detected in {image_path}"
        
        # Recognition
        face_crop = extract_face_crop(image, detection_result.faces[0].bbox)
        recognition_result = await recognition_service.recognize_face(face_crop)
        
        # ตรวจสอบผลลัพธ์
        if expected_identity == "unknown":
            assert recognition_result.best_match is None, f"Unknown face wrongly recognized"
        else:
            assert recognition_result.best_match is not None, f"Known face not recognized"
            assert recognition_result.best_match.person_id == expected_identity, f"Wrong identity"
            assert recognition_result.confidence >= min_confidence, f"Confidence too low: {recognition_result.confidence}"
        
        print(f"✅ {image_path}: {recognition_result.best_match.person_id if recognition_result.best_match else 'unknown'} ({recognition_result.confidence:.3f})")
```

#### **2.2 ทดสอบ Performance Benchmarks**
```python
async def test_performance_benchmarks():
    """ทดสอบประสิทธิภาพ"""
    service = FaceAnalysisService(vram_manager, config)
    await service.initialize()
    
    # โหลดภาพทดสอบ
    test_images = load_test_images("test_images")
    
    processing_times = []
    accuracy_scores = []
    
    for image_path in test_images:
        start_time = time.time()
        
        image = cv2.imread(image_path)
        result = await service.analyze_faces(image, config)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # คำนวณ accuracy (ถ้ามี ground truth)
        if has_ground_truth(image_path):
            accuracy = calculate_accuracy(result, get_ground_truth(image_path))
            accuracy_scores.append(accuracy)
    
    # ตรวจสอบเกณฑ์
    avg_time = np.mean(processing_times)
    avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
    
    assert avg_time < 2.0, f"Processing too slow: {avg_time:.3f}s"
    assert avg_accuracy > 0.85, f"Accuracy too low: {avg_accuracy:.3f}"
    
    print(f"✅ Performance: {avg_time:.3f}s/image, {avg_accuracy:.1%} accuracy")
```

### **Step 3: Real-world Testing - สภาพจริง**

#### **3.1 ทดสอบกับภาพใหม่**
```python
async def test_with_new_images():
    """ทดสอบกับภาพที่ไม่เคยเห็น"""
    # ถ่ายภาพใหม่ของ boss และ night
    new_test_images = [
        "new_images/boss_new_01.jpg",
        "new_images/boss_new_02.jpg", 
        "new_images/night_new_01.jpg",
        "new_images/night_new_02.jpg",
        "new_images/stranger_01.jpg",
        "new_images/stranger_02.jpg"
    ]
    
    expected_results = {
        "boss_new": "boss",
        "night_new": "night", 
        "stranger": "unknown"
    }
    
    correct_predictions = 0
    total_predictions = 0
    
    for image_path in new_test_images:
        result = await analyze_single_image(image_path)
        
        # ดึง expected result
        for key, expected in expected_results.items():
            if key in image_path:
                if result.identity == expected:
                    correct_predictions += 1
                total_predictions += 1
                break
    
    accuracy = correct_predictions / total_predictions
    print(f"📊 New images accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
```

#### **3.2 ทดสอบ Edge Cases**
```python
async def test_edge_cases():
    """ทดสอบกรณีพิเศษ"""
    edge_cases = [
        ("very_dark_image.jpg", "low lighting"),
        ("very_bright_image.jpg", "overexposed"),
        ("blurry_image.jpg", "motion blur"),
        ("low_resolution.jpg", "low quality"),
        ("multiple_faces.jpg", "group photo"),
        ("partial_face.jpg", "face cut off"),
        ("side_profile.jpg", "non-frontal"),
        ("with_mask.jpg", "face mask"),
        ("with_sunglasses.jpg", "occlusion")
    ]
    
    results = {}
    
    for image_file, condition in edge_cases:
        try:
            result = await analyze_single_image(f"edge_cases/{image_file}")
            
            results[condition] = {
                'success': True,
                'faces_detected': len(result.faces),
                'avg_quality': np.mean([f.quality_score for f in result.faces]) if result.faces else 0,
                'processing_time': result.total_time
            }
            
        except Exception as e:
            results[condition] = {
                'success': False,
                'error': str(e)
            }
    
    # รายงานผล
    for condition, result in results.items():
        if result['success']:
            print(f"✅ {condition}: {result['faces_detected']} faces, quality {result['avg_quality']:.1f}")
        else:
            print(f"❌ {condition}: {result['error']}")
```

---

## **เกณฑ์การผ่านการทดสอบ**

### **ระดับ Critical (ต้องผ่าน 100%)**
- ✅ boss_04.jpg confidence >95%
- ✅ Quality scores อยู่ในช่วง 0-100
- ✅ ไม่มี bbox ที่ครอบคลุมทั้งภาพ (area_ratio <90%)
- ✅ Unknown faces ถูกจำแนกถูกต้อง >80%

### **ระดับ High (ต้องผ่าน >90%)**
- ✅ Overall recognition accuracy >85%
- ✅ Processing time <2s per image
- ✅ False positive rate <10%
- ✅ System stability (no crashes)

### **ระดับ Medium (ต้องผ่าน >70%)**
- ✅ Edge cases handling >70%
- ✅ Multi-face scenarios >80%
- ✅ Different lighting conditions >75%

---

## **การวัดและรายงานผล**

### **Metrics ที่ต้องเก็บ**
```python
test_metrics = {
    'accuracy': {
        'boss_recognition': 0.0,  # อัตราการจำแนก boss ถูกต้อง
        'night_recognition': 0.0, # อัตราการจำแนก night ถูกต้อง
        'unknown_detection': 0.0,  # อัตราการจำแนก unknown ถูกต้อง
        'overall_accuracy': 0.0
    },
    'performance': {
        'avg_processing_time': 0.0,
        'detection_time': 0.0,
        'recognition_time': 0.0,
        'throughput_fps': 0.0
    },
    'quality': {
        'avg_detection_quality': 0.0,
        'valid_detection_ratio': 0.0,
        'bbox_accuracy': 0.0
    },
    'robustness': {
        'edge_case_success_rate': 0.0,
        'error_rate': 0.0,
        'system_stability': 0.0
    }
}
```

### **รายงานผลแบบละเอียด**
```python
def generate_test_report(metrics, test_results):
    report = f"""
# 📋 Face Recognition System Test Report

## 🎯 การแก้ไขปัญหาหลัก
- boss_04.jpg confidence: {test_results['boss_04_confidence']:.1%} (เป้าหมาย: >95%)
- Quality score range: ✅ All within 0-100
- Bbox accuracy: {metrics['quality']['bbox_accuracy']:.1%}
- Unknown detection: {metrics['accuracy']['unknown_detection']:.1%}

## 📊 ประสิทธิภาพโดยรวม
- Overall accuracy: {metrics['accuracy']['overall_accuracy']:.1%}
- Processing time: {metrics['performance']['avg_processing_time']:.3f}s
- Detection quality: {metrics['quality']['avg_detection_quality']:.1f}/100
- System stability: {metrics['robustness']['system_stability']:.1%}

## 🔍 การทดสอบแต่ละส่วน
- Boss recognition: {metrics['accuracy']['boss_recognition']:.1%}
- Night recognition: {metrics['accuracy']['night_recognition']:.1%}
- Unknown detection: {metrics['accuracy']['unknown_detection']:.1%}
- Edge cases: {metrics['robustness']['edge_case_success_rate']:.1%}

## ✅ สรุปผลการทดสอบ
- Critical tests: {'PASS' if all_critical_passed else 'FAIL'}
- Performance tests: {'PASS' if performance_passed else 'FAIL'}
- Robustness tests: {'PASS' if robustness_passed else 'FAIL'}

Overall: {'PASS' if overall_passed else 'FAIL'}
"""
    return report
```

---

## **แผนการ Deploy และ Monitor**

### **1. Gradual Rollout**
- Week 1: Unit testing และแก้ไขปัญหาที่พบ
- Week 2: Integration testing และ performance tuning  
- Week 3: Real-world testing และ edge case handling
- Week 4: Production deployment และ monitoring

### **2. Continuous Monitoring**
```python
# Production monitoring metrics
production_metrics = {
    'daily_accuracy': [],
    'processing_times': [],
    'error_rates': [],
    'unknown_detection_rates': [],
    'user_feedback': []
}

# Alert thresholds
alert_thresholds = {
    'accuracy_drop': 0.10,  # แจ้งเตือนถ้า accuracy ลด >10%
    'slow_processing': 3.0,  # แจ้งเตือนถ้าช้า >3 วินาที
    'high_error_rate': 0.05  # แจ้งเตือนถ้า error >5%
}
```

การทดสอบแบบละเอียดนี้จะช่วยให้มั่นใจได้ว่าระบบที่แก้ไขแล้วจะทำงานได้ถูกต้องและมีประสิทธิภาพดีในสภาพการใช้งานจริง!
## 📋 **สรุปภาพรวมและลำดับความสำคัญ**

### **🚨 ปัญหาร้ายแรงที่ต้องแก้ทันที**

| ปัญหา | ระดับความร้ายแรง | สาเหตุ | ผลกระทบ |
|-------|------------------|--------|----------|
| **ใช้ Random Embedding** | 🔴 Critical | `np.random.randn()` แทนโมเดลจริง | boss_04 ได้แค่ 51.8% แทนที่จะเป็น >95% |
| **ไม่มี Unknown Detection** | 🔴 Critical | ไม่มี threshold checking | จำแนกใบหน้าแปลกหน้าผิด 100% |
| **Quality Score ผิดปกติ** | 🟠 High | คำนวณผิด คืนค่า >100 | ระบบประเมินคุณภาพผิด |
| **Bbox ครอบคลุมทั้งภาพ** | 🟠 High | Detection model parameters | ตัดใบหน้าไม่แม่นยำ |

### **🔧 โซลูชันที่เสนอ**

#### **Phase 1: Core Fixes (Priority 1)**
1. **✅ เปลี่ยนเป็น Real ONNX Models** - ใช้โมเดลจริง แทน fake embedding
2. **✅ เพิ่ม Unknown Detection Logic** - threshold checking เพื่อจำแนก unknown
3. **✅ แก้ไข Quality Calculation** - ให้คืนค่า 0-100 เท่านั้น
4. **✅ ปรับ Detection Parameters** - เพิ่ม confidence threshold เป็น 0.4

#### **Phase 2: Enhanced Features (Priority 2)**  
1. **✅ Multi-Embedding Gallery** - รองรับหลาย embeddings ต่อคน
2. **✅ Intelligent Model Selection** - เลือกโมเดลตามสถานการณ์
3. **✅ Quality-based Filtering** - กรองใบหน้าคุณภาพต่ำ
4. **✅ Performance Monitoring** - ติดตามประสิทธิภาพ

### **🎯 ผลลัพธ์ที่คาดหวัง**

#### **Before (ปัจจุบัน)**
```json
"boss_04.jpg": {
  "confidence": 0.518,  // ❌ แค่ 51.8%
  "quality_score": 789,  // ❌ เกิน 100
  "best_match": "boss"   // ❌ ถูกโดยบังเอิญ
}
```

#### **After (หลังแก้ไข)**
```json
"boss_04.jpg": {
  "confidence": 0.96,   // ✅ >95%
  "quality_score": 87,  // ✅ อยู่ใน 0-100
  "best_match": "boss"  // ✅ ถูกต้องจากโมเดลจริง
}

"unknown_person.jpg": {
  "confidence": 0.0,    // ✅ จำแนก unknown ได้
  "best_match": null    // ✅ ไม่จำผิด
}
```

### **📈 เกณฑ์ความสำเร็จ**

| Metric | Current | Target | Critical |
|--------|---------|--------|----------|
| boss_04 Confidence | 51.8% | >95% | ✅ |
| Unknown Detection | 0% | >80% | ✅ |
| Quality Score Range | 0-789 | 0-100 | ✅ |
| Processing Time | 0.67s | <2s | ✅ |
| Overall Accuracy | ~76% | >85% | 🎯 |

### **⚡ การดำเนินการทันที**

#### **Step 1: แทนที่ Core Service (Day 1-3)**
1. Replace `face_recognition_service.py` ด้วย **Real ONNX Implementation**
2. Update `utils.py` ด้วย **Fixed Quality Calculation**  
3. Test กับ `boss_04.jpg` - ต้องได้ >95%

#### **Step 2: Deploy และ Test (Day 4-7)**
1. Run `multi_embedding_face_test.py` ใหม่
2. ตรวจสอบผลลัพธ์ตาม **Testing Plan**
3. Fine-tune thresholds ถ้าจำเป็น

#### **Step 3: Production Ready (Week 2)**
1. Complete integration testing
2. Performance optimization
3. Deploy to production

---

## 🎊 **สรุป: จากปัญหาสู่โซลูชัน**

**ปัญหาหลัก:** ระบบใช้ **Random Numbers แทนโมเดล AI จริง** ทำให้ผลลัพธ์ไม่น่าเชื่อถือ

**โซลูชัน:** เปลี่ยนเป็น **Real ONNX Models + Unknown Detection + Quality Fixes**

**ผลลัพธ์ที่คาดหวัง:** 
- ✅ boss_04.jpg จาก 51.8% → >95%
- ✅ Unknown faces จาก 0% → >80% detection rate  
- ✅ System reliability จาก Poor → Excellent

**การลงทุน:** ~1-2 สัปดาห์ในการแก้ไข + testing

**Return on Investment:** ระบบจะใช้งานได้จริงในระดับ Production แทนที่จะเป็นเพียง Demo เท่านั้น!