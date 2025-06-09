# Face Detection Service - Multi-Engine Implementation

## Comprehensive Face Detection with YOLO, MediaPipe, and InsightFace

This document provides detailed specifications for the face detection service supporting three detection engines: YOLO (primary GPU), MediaPipe (CPU fallback), and InsightFace (advanced features).

---

## 🎯 Service Overview

### Detection Engine Comparison

| Engine | Device | Speed | Accuracy | Features | Use Case |
|--------|--------|--------|----------|----------|----------|
| **YOLO v10n** | GPU | Very Fast | High | Bounding boxes, confidence | Real-time primary |
| **YOLO v5s** | GPU | Fast | Very High | Bounding boxes, landmarks | High accuracy needs |
| **MediaPipe** | CPU | Moderate | Good | Landmarks, mesh | CPU fallback |
| **InsightFace** | GPU/CPU | Moderate | Excellent | Full analysis suite | Advanced features |

### Performance Characteristics

```yaml
YOLO v10n (Primary):
  VRAM Usage: 9MB
  Processing Speed: 50-100 FPS (GPU)
  Accuracy: 95%+ on standard datasets
  Features: Bounding boxes, confidence scores
  
YOLO v5s (Secondary):
  VRAM Usage: 28MB  
  Processing Speed: 30-60 FPS (GPU)
  Accuracy: 97%+ on standard datasets
  Features: Bounding boxes, basic landmarks
  
MediaPipe (Fallback):
  CPU Usage: 1-2 cores
  Processing Speed: 15-30 FPS (CPU)
  Accuracy: 90%+ on standard datasets
  Features: 468 face landmarks, mesh
  
InsightFace (Advanced):
  VRAM Usage: 15MB (detection model)
  Processing Speed: 20-40 FPS (GPU)
  Accuracy: 98%+ on standard datasets
  Features: Detection, landmarks, attributes
```

---

## 🏗️ Architecture Implementation

### Service Interface Design

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

class DetectionEngine(Enum):
    YOLO_V10N = "yolo_v10n"
    YOLO_V5S = "yolo_v5s"
    MEDIAPIPE = "mediapipe"
    INSIGHTFACE = "insightface"

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

@dataclass
class FaceLandmarks:
    points: List[tuple]  # [(x, y), ...]
    confidence: Optional[float] = None
    landmark_type: str = "68_point"  # 68_point, 5_point, 468_point

@dataclass
class FaceDetection:
    bbox: BoundingBox
    landmarks: Optional[FaceLandmarks] = None
    attributes: Optional[Dict[str, Any]] = None
    engine_used: str = ""
    processing_time: float = 0.0

@dataclass
class DetectionResult:
    faces: List[FaceDetection]
    image_shape: tuple
    total_processing_time: float
    engine_used: str
    fallback_used: bool = False
    error: Optional[str] = None

class FaceDetectionEngine(ABC):
    """Base interface for all face detection engines"""
    
    @abstractmethod
    async def initialize(self, device: str = "cuda") -> bool:
        """Initialize the detection engine"""
        pass
    
    @abstractmethod
    async def detect_faces(self, 
                          image: np.ndarray,
                          confidence_threshold: float = 0.5,
                          **kwargs) -> DetectionResult:
        """Detect faces in image"""
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine metadata and capabilities"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Clean up resources"""
        pass

class FaceDetectionService:
    """
    Multi-engine face detection service with intelligent fallback
    """
    
    def __init__(self, vram_manager, config: Dict[str, Any]):
        self.vram_manager = vram_manager
        self.config = config
        self.engines = {}
        self.primary_engine = DetectionEngine.YOLO_V10N
        self.fallback_engine = DetectionEngine.MEDIAPIPE
        self.performance_tracker = PerformanceTracker()
        
    async def initialize(self):
        """Initialize all available detection engines"""
        # Initialize YOLO engines
        self.engines[DetectionEngine.YOLO_V10N] = YOLOv10nEngine(
            model_path="models/face-detection/yolov10n-face.onnx",
            vram_manager=self.vram_manager
        )
        
        self.engines[DetectionEngine.YOLO_V5S] = YOLOv5sEngine(
            model_path="models/face-detection/yolov5s-face.onnx",
            vram_manager=self.vram_manager
        )
        
        # Initialize MediaPipe (CPU fallback)
        self.engines[DetectionEngine.MEDIAPIPE] = MediaPipeEngine()
        
        # Initialize InsightFace (advanced features)
        self.engines[DetectionEngine.INSIGHTFACE] = InsightFaceEngine(
            vram_manager=self.vram_manager
        )
        
        # Initialize primary engine
        await self.engines[self.primary_engine].initialize("cuda")
        
        # Initialize fallback engine
        await self.engines[self.fallback_engine].initialize("cpu")
    
    async def detect_faces(self, 
                          image: np.ndarray,
                          engine: Optional[DetectionEngine] = None,
                          confidence_threshold: float = 0.5,
                          return_landmarks: bool = False,
                          return_attributes: bool = False,
                          **kwargs) -> DetectionResult:
        """
        Detect faces with automatic engine selection and fallback
        """
        start_time = time.time()
        
        # Determine which engine to use
        selected_engine = engine or await self._select_optimal_engine(
            image.shape, return_landmarks, return_attributes
        )
        
        try:
            # Attempt detection with selected engine
            result = await self._detect_with_engine(
                image, selected_engine, confidence_threshold, 
                return_landmarks, return_attributes, **kwargs
            )
            
            if result.error is None:
                # Record successful detection
                self.performance_tracker.record_detection(
                    engine=selected_engine.value,
                    processing_time=result.total_processing_time,
                    face_count=len(result.faces),
                    success=True
                )
                return result
                
        except Exception as e:
            logger.error(f"Detection failed with {selected_engine.value}: {e}")
        
        # Fallback to CPU if primary failed
        if selected_engine != self.fallback_engine:
            logger.info(f"Falling back to {self.fallback_engine.value}")
            
            try:
                result = await self._detect_with_engine(
                    image, self.fallback_engine, confidence_threshold,
                    return_landmarks, return_attributes, **kwargs
                )
                result.fallback_used = True
                
                self.performance_tracker.record_detection(
                    engine=self.fallback_engine.value,
                    processing_time=result.total_processing_time,
                    face_count=len(result.faces),
                    success=True,
                    fallback=True
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Fallback detection also failed: {e}")
        
        # Return error result
        total_time = time.time() - start_time
        return DetectionResult(
            faces=[],
            image_shape=image.shape,
            total_processing_time=total_time,
            engine_used="none",
            error="All detection engines failed"
        )
```

---

## 🤖 YOLO Implementation

### YOLO v10n Engine (Primary)

```python
class YOLOv10nEngine(FaceDetectionEngine):
    """
    YOLO v10n face detection - optimized for speed and VRAM efficiency
    """
    
    def __init__(self, model_path: str, vram_manager):
        self.model_path = model_path
        self.vram_manager = vram_manager
        self.session = None
        self.input_size = (640, 640)
        self.model_loaded = False
        
    async def initialize(self, device: str = "cuda") -> bool:
        """Initialize YOLO v10n model"""
        try:
            if device == "cuda":
                # Request VRAM allocation
                allocation = await self.vram_manager.request_model_allocation(
                    "yolov10n-face", "critical", "face_detection_service"
                )
                
                if allocation.location == "gpu":
                    providers = ['CUDAExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                    device = "cpu"
            else:
                providers = ['CPUExecutionProvider']
            
            # Create ONNX session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            if device == "cuda":
                # CUDA provider options
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 512 * 1024 * 1024,  # 512MB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                providers = [('CUDAExecutionProvider', cuda_options)]
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Get model info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.model_loaded = True
            self.device = device
            
            logger.info(f"YOLO v10n initialized on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO v10n: {e}")
            return False
    
    async def detect_faces(self, 
                          image: np.ndarray,
                          confidence_threshold: float = 0.5,
                          **kwargs) -> DetectionResult:
        """Detect faces using YOLO v10n"""
        if not self.model_loaded:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            input_tensor, scale_factors = self._preprocess_image(image)
            
            # Run inference
            inference_start = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            inference_time = time.time() - inference_start
            
            # Post-process results
            detections = self._postprocess_outputs(
                outputs, scale_factors, confidence_threshold
            )
            
            total_time = time.time() - start_time
            
            return DetectionResult(
                faces=detections,
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="yolo_v10n"
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"YOLO v10n detection failed: {e}")
            return DetectionResult(
                faces=[],
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="yolo_v10n",
                error=str(e)
            )
    
    def _preprocess_image(self, image: np.ndarray) -> tuple:
        """Preprocess image for YOLO inference"""
        original_height, original_width = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        target_height, target_width = self.input_size
        scale = min(target_width / original_width, target_height / original_height)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Pad to target size
        padded_image = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        padded_image[y_offset:y_offset + new_height, 
                    x_offset:x_offset + new_width] = resized_image
        
        # Convert to CHW format and normalize
        input_tensor = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        scale_factors = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'original_width': original_width,
            'original_height': original_height
        }
        
        return input_tensor, scale_factors
    
    def _postprocess_outputs(self, 
                           outputs: List[np.ndarray], 
                           scale_factors: Dict[str, Any],
                           confidence_threshold: float) -> List[FaceDetection]:
        """Post-process YOLO outputs to face detections"""
        detections = []
        
        # YOLO v10n output format: [batch, anchors, 5 + num_classes]
        # 5 = x, y, w, h, objectness
        predictions = outputs[0][0]  # Remove batch dimension
        
        for prediction in predictions:
            # Extract box coordinates and confidence
            x_center, y_center, width, height, objectness = prediction[:5]
            
            if objectness < confidence_threshold:
                continue
            
            # Convert from center format to corner format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Scale back to original image coordinates
            scale = scale_factors['scale']
            x_offset = scale_factors['x_offset']
            y_offset = scale_factors['y_offset']
            
            x1 = (x1 * self.input_size[1] - x_offset) / scale
            y1 = (y1 * self.input_size[0] - y_offset) / scale
            x2 = (x2 * self.input_size[1] - x_offset) / scale
            y2 = (y2 * self.input_size[0] - y_offset) / scale
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, scale_factors['original_width']))
            y1 = max(0, min(y1, scale_factors['original_height']))
            x2 = max(0, min(x2, scale_factors['original_width']))
            y2 = max(0, min(y2, scale_factors['original_height']))
            
            # Create detection object
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=objectness)
            
            detection = FaceDetection(
                bbox=bbox,
                engine_used="yolo_v10n"
            )
            
            detections.append(detection)
        
        # Apply Non-Maximum Suppression
        detections = self._apply_nms(detections, iou_threshold=0.5)
        
        return detections
    
    def _apply_nms(self, 
                   detections: List[FaceDetection], 
                   iou_threshold: float = 0.5) -> List[FaceDetection]:
        """Apply Non-Maximum Suppression to filter overlapping detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x.bbox.confidence, reverse=True)
        
        filtered_detections = []
        
        for i, detection in enumerate(detections):
            is_suppressed = False
            
            for kept_detection in filtered_detections:
                iou = self._calculate_iou(detection.bbox, kept_detection.bbox)
                if iou > iou_threshold:
                    is_suppressed = True
                    break
            
            if not is_suppressed:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Calculate intersection area
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
```

### YOLO v5s Engine (High Accuracy)

```python
class YOLOv5sEngine(FaceDetectionEngine):
    """
    YOLO v5s face detection - optimized for accuracy with basic landmarks
    """
    
    def __init__(self, model_path: str, vram_manager):
        self.model_path = model_path
        self.vram_manager = vram_manager
        self.session = None
        self.input_size = (640, 640)
        self.model_loaded = False
        
    async def initialize(self, device: str = "cuda") -> bool:
        """Initialize YOLO v5s model"""
        try:
            if device == "cuda":
                allocation = await self.vram_manager.request_model_allocation(
                    "yolov5s-face", "low", "face_detection_service"
                )
                
                if allocation.location == "cpu":
                    device = "cpu"
            
            # Similar initialization to v10n but with different model
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = self._get_providers(device)
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.model_loaded = True
            self.device = device
            
            logger.info(f"YOLO v5s initialized on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO v5s: {e}")
            return False
    
    async def detect_faces(self, 
                          image: np.ndarray,
                          confidence_threshold: float = 0.5,
                          return_landmarks: bool = False,
                          **kwargs) -> DetectionResult:
        """Detect faces using YOLO v5s with optional landmarks"""
        if not self.model_loaded:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Similar preprocessing to v10n
            input_tensor, scale_factors = self._preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Post-process with landmarks support
            detections = self._postprocess_outputs_with_landmarks(
                outputs, scale_factors, confidence_threshold, return_landmarks
            )
            
            total_time = time.time() - start_time
            
            return DetectionResult(
                faces=detections,
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="yolo_v5s"
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            return DetectionResult(
                faces=[],
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="yolo_v5s",
                error=str(e)
            )
    
    def _postprocess_outputs_with_landmarks(self, 
                                           outputs: List[np.ndarray], 
                                           scale_factors: Dict[str, Any],
                                           confidence_threshold: float,
                                           return_landmarks: bool) -> List[FaceDetection]:
        """Post-process YOLO v5s outputs with landmark support"""
        detections = []
        predictions = outputs[0][0]
        
        for prediction in predictions:
            # YOLO v5s face format: [x, y, w, h, conf, cls, landmarks...]
            if len(prediction) >= 6:
                x_center, y_center, width, height, conf, cls = prediction[:6]
                
                if conf < confidence_threshold:
                    continue
                
                # Convert coordinates
                x1, y1, x2, y2 = self._convert_coordinates(
                    x_center, y_center, width, height, scale_factors
                )
                
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                
                # Extract landmarks if available and requested
                landmarks = None
                if return_landmarks and len(prediction) > 6:
                    landmarks = self._extract_landmarks(
                        prediction[6:], scale_factors
                    )
                
                detection = FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    engine_used="yolo_v5s"
                )
                
                detections.append(detection)
        
        return self._apply_nms(detections)
    
    def _extract_landmarks(self, 
                          landmark_data: np.ndarray, 
                          scale_factors: Dict[str, Any]) -> FaceLandmarks:
        """Extract and scale landmarks from YOLO v5s output"""
        # YOLO v5s typically provides 5-point landmarks
        # Format: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        landmarks = []
        
        for i in range(0, len(landmark_data), 2):
            if i + 1 < len(landmark_data):
                x = landmark_data[i]
                y = landmark_data[i + 1]
                
                # Scale back to original coordinates
                scale = scale_factors['scale']
                x_offset = scale_factors['x_offset']
                y_offset = scale_factors['y_offset']
                
                scaled_x = (x * self.input_size[1] - x_offset) / scale
                scaled_y = (y * self.input_size[0] - y_offset) / scale
                
                landmarks.append((scaled_x, scaled_y))
        
        return FaceLandmarks(
            points=landmarks,
            landmark_type="5_point"
        )
```

---

## 📱 MediaPipe Implementation (CPU Fallback)

```python
class MediaPipeEngine(FaceDetectionEngine):
    """
    MediaPipe face detection - CPU fallback with comprehensive landmarks
    """
    
    def __init__(self):
        self.face_detection = None
        self.face_mesh = None
        self.model_loaded = False
        
    async def initialize(self, device: str = "cpu") -> bool:
        """Initialize MediaPipe models"""
        try:
            import mediapipe as mp
            
            # Face detection model
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            
            # Face mesh for landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.model_loaded = True
            logger.info("MediaPipe initialized on CPU")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False
    
    async def detect_faces(self, 
                          image: np.ndarray,
                          confidence_threshold: float = 0.5,
                          return_landmarks: bool = False,
                          **kwargs) -> DetectionResult:
        """Detect faces using MediaPipe"""
        if not self.model_loaded:
            raise RuntimeError("MediaPipe not initialized")
        
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            detections = []
            
            # Face detection
            detection_results = self.face_detection.process(rgb_image)
            
            if detection_results.detections:
                for detection in detection_results.detections:
                    # Extract bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    x1 = bbox.xmin * width
                    y1 = bbox.ymin * height
                    x2 = (bbox.xmin + bbox.width) * width
                    y2 = (bbox.ymin + bbox.height) * height
                    
                    confidence = detection.score[0]
                    
                    if confidence < confidence_threshold:
                        continue
                    
                    bbox_obj = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2, 
                        confidence=confidence
                    )
                    
                    # Get landmarks if requested
                    landmarks = None
                    if return_landmarks:
                        landmarks = await self._get_landmarks(rgb_image, bbox_obj)
                    
                    face_detection = FaceDetection(
                        bbox=bbox_obj,
                        landmarks=landmarks,
                        engine_used="mediapipe"
                    )
                    
                    detections.append(face_detection)
            
            total_time = time.time() - start_time
            
            return DetectionResult(
                faces=detections,
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="mediapipe"
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            return DetectionResult(
                faces=[],
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="mediapipe",
                error=str(e)
            )
    
    async def _get_landmarks(self, 
                           rgb_image: np.ndarray, 
                           bbox: BoundingBox) -> FaceLandmarks:
        """Extract detailed landmarks using MediaPipe Face Mesh"""
        try:
            mesh_results = self.face_mesh.process(rgb_image)
            
            if mesh_results.multi_face_landmarks:
                height, width = rgb_image.shape[:2]
                
                # Find landmarks within bounding box
                for face_landmarks in mesh_results.multi_face_landmarks:
                    landmarks = []
                    
                    # Convert normalized coordinates to pixel coordinates
                    for landmark in face_landmarks.landmark:
                        x = landmark.x * width
                        y = landmark.y * height
                        
                        # Check if landmark is within bounding box
                        if (bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2):
                            landmarks.append((x, y))
                    
                    if landmarks:
                        return FaceLandmarks(
                            points=landmarks,
                            landmark_type="468_point"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Landmark extraction failed: {e}")
            return None
```

---

## 🔍 InsightFace Implementation (Advanced Features)

```python
class InsightFaceEngine(FaceDetectionEngine):
    """
    InsightFace detection with advanced analysis capabilities
    """
    
    def __init__(self, vram_manager):
        self.vram_manager = vram_manager
        self.app = None
        self.model_loaded = False
        
    async def initialize(self, device: str = "cuda") -> bool:
        """Initialize InsightFace models"""
        try:
            import insightface
            
            if device == "cuda":
                allocation = await self.vram_manager.request_model_allocation(
                    "insightface-detection", "medium", "face_detection_service"
                )
                
                if allocation.location == "cpu":
                    device = "cpu"
            
            # Initialize InsightFace app
            ctx_id = 0 if device == "cuda" else -1
            
            self.app = insightface.app.FaceAnalysis(
                providers=['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            self.model_loaded = True
            self.device = device
            
            logger.info(f"InsightFace initialized on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            return False
    
    async def detect_faces(self, 
                          image: np.ndarray,
                          confidence_threshold: float = 0.5,
                          return_landmarks: bool = False,
                          return_attributes: bool = False,
                          **kwargs) -> DetectionResult:
        """Detect faces using InsightFace with advanced features"""
        if not self.model_loaded:
            raise RuntimeError("InsightFace not initialized")
        
        start_time = time.time()
        
        try:
            # Run InsightFace analysis
            faces = self.app.get(image)
            
            detections = []
            
            for face in faces:
                # Extract bounding box
                bbox = face.bbox
                confidence = face.det_score
                
                if confidence < confidence_threshold:
                    continue
                
                bbox_obj = BoundingBox(
                    x1=bbox[0], y1=bbox[1], 
                    x2=bbox[2], y2=bbox[3], 
                    confidence=confidence
                )
                
                # Extract landmarks
                landmarks = None
                if return_landmarks and hasattr(face, 'kps'):
                    landmark_points = [(kp[0], kp[1]) for kp in face.kps]
                    landmarks = FaceLandmarks(
                        points=landmark_points,
                        landmark_type="5_point"
                    )
                
                # Extract attributes
                attributes = {}
                if return_attributes:
                    if hasattr(face, 'age'):
                        attributes['age'] = face.age
                    if hasattr(face, 'gender'):
                        attributes['gender'] = face.gender
                    if hasattr(face, 'embedding'):
                        attributes['embedding_available'] = True
                
                detection = FaceDetection(
                    bbox=bbox_obj,
                    landmarks=landmarks,
                    attributes=attributes if attributes else None,
                    engine_used="insightface"
                )
                
                detections.append(detection)
            
            total_time = time.time() - start_time
            
            return DetectionResult(
                faces=detections,
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="insightface"
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            return DetectionResult(
                faces=[],
                image_shape=image.shape,
                total_processing_time=total_time,
                engine_used="insightface",
                error=str(e)
            )
```

---

## 🎛️ Engine Selection Logic

```python
class EngineSelector:
    """
    Intelligent engine selection based on requirements and availability
    """
    
    def __init__(self, face_detection_service):
        self.service = face_detection_service
        self.performance_history = {}
        
    async def select_optimal_engine(self, 
                                   image_shape: tuple,
                                   return_landmarks: bool = False,
                                   return_attributes: bool = False,
                                   priority: str = "speed") -> DetectionEngine:
        """
        Select optimal detection engine based on requirements
        """
        # Advanced features require specific engines
        if return_attributes:
            if await self._is_engine_available(DetectionEngine.INSIGHTFACE):
                return DetectionEngine.INSIGHTFACE
        
        # High-quality landmarks prefer MediaPipe or InsightFace
        if return_landmarks:
            if priority == "accuracy":
                if await self._is_engine_available(DetectionEngine.INSIGHTFACE):
                    return DetectionEngine.INSIGHTFACE
                elif await self._is_engine_available(DetectionEngine.MEDIAPIPE):
                    return DetectionEngine.MEDIAPIPE
        
        # Speed-optimized selection
        if priority == "speed":
            # Check VRAM availability for GPU engines
            vram_status = await self.service.vram_manager.get_memory_status()
            
            if vram_status.available_mb > 50:  # Enough for YOLO v10n
                if await self._is_engine_available(DetectionEngine.YOLO_V10N):
                    return DetectionEngine.YOLO_V10N
            
            if vram_status.available_mb > 150:  # Enough for YOLO v5s
                if await self._is_engine_available(DetectionEngine.YOLO_V5S):
                    return DetectionEngine.YOLO_V5S
        
        # Accuracy-optimized selection
        if priority == "accuracy":
            if await self._is_engine_available(DetectionEngine.YOLO_V5S):
                return DetectionEngine.YOLO_V5S
            elif await self._is_engine_available(DetectionEngine.INSIGHTFACE):
                return DetectionEngine.INSIGHTFACE
        
        # Fallback to MediaPipe (always available on CPU)
        return DetectionEngine.MEDIAPIPE
    
    async def _is_engine_available(self, engine: DetectionEngine) -> bool:
        """Check if engine is available and functional"""
        if engine not in self.service.engines:
            return False
        
        engine_obj = self.service.engines[engine]
        return engine_obj.model_loaded
```

---

*This comprehensive face detection service provides robust multi-engine support with intelligent fallback mechanisms, ensuring reliable face detection across different performance requirements and hardware constraints.*
