# Deepfake Detection Service

## Overview

The Deepfake Detection Service provides advanced AI-generated content detection capabilities to identify synthetic faces, manipulated videos, and sophisticated deepfake attacks. This service is crucial for maintaining content authenticity and preventing malicious use of synthetic media.

## Model Architecture

### Supported Models

| Model | VRAM Usage | Accuracy | Speed | Detection Type |
|-------|------------|----------|-------|----------------|
| **EfficientNet-B4 Deepfake** | 44MB | 95.8% | 35ms | General deepfake detection |
| **XceptionNet Deepfake** | 88MB | 97.2% | 45ms | High-accuracy detection |
| **MobileNet Deepfake** | 15MB | 92.1% | 20ms | Lightweight detection |

### Detection Capabilities

```python
class DeepfakeDetectionTypes:
    FACE_SWAP = "face_swap"           # Face replacement deepfakes
    FACE_REENACTMENT = "reenactment" # Expression/pose manipulation
    SPEECH_DRIVEN = "speech_driven"   # Audio-driven animation
    FULL_SYNTHESIS = "full_synthesis" # Completely synthetic faces
    PARTIAL_MANIPULATION = "partial"  # Localized face edits
    AUTHENTIC = "authentic"           # Real, unmanipulated content
    UNKNOWN = "unknown"               # Unable to classify
    
    @classmethod
    def get_manipulation_types(cls):
        return [cls.FACE_SWAP, cls.FACE_REENACTMENT, cls.SPEECH_DRIVEN, 
                cls.FULL_SYNTHESIS, cls.PARTIAL_MANIPULATION]
```

## Implementation

### Core Service Class

```python
import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
import time
import asyncio
from sklearn.metrics import roc_auc_score

@dataclass
class DeepfakeDetectionResult:
    """Deepfake detection result"""
    is_authentic: bool
    confidence: float
    manipulation_type: str
    authenticity_score: float
    processing_time: float
    model_used: str
    face_quality: float
    temporal_consistency: Optional[float] = None
    artifacts_detected: List[str] = None

@dataclass
class TemporalAnalysis:
    """Temporal analysis for video sequences"""
    frame_consistency: float
    motion_patterns: float
    lighting_consistency: float
    compression_artifacts: float

class DeepfakeDetectionService:
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.current_model = 'efficientnet_b4'
        self.logger = logging.getLogger(__name__)
        
        # Detection thresholds for different models
        self.thresholds = {
            'efficientnet_b4': {
                'authenticity': 0.5,
                'confidence': 0.7
            },
            'xceptionnet': {
                'authenticity': 0.4,
                'confidence': 0.75
            },
            'mobilenet': {
                'authenticity': 0.6,
                'confidence': 0.65
            }
        }
        
        # Performance statistics
        self.stats = {
            'total_detections': 0,
            'authentic_faces': 0,
            'deepfake_faces': 0,
            'manipulation_types': {},
            'average_processing_time': 0,
            'model_usage': {},
            'accuracy_metrics': {}
        }
        
        # Artifact detection patterns
        self.artifact_detectors = self._initialize_artifact_detectors()
    
    async def initialize(self, model_name: str = None):
        """Initialize the deepfake detection service"""
        try:
            if model_name:
                await self._load_model(model_name)
            else:
                await self._load_model('efficientnet_b4')  # Default model
            
            self.logger.info(f"Deepfake Detection Service initialized with {self.current_model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Deepfake Detection Service: {e}")
            return False
    
    async def _load_model(self, model_name: str):
        """Load specific deepfake detection model"""
        model_paths = {
            'efficientnet_b4': 'models/deepfake-detection/efficientnet_b4_deepfake.onnx',
            'xceptionnet': 'models/deepfake-detection/xceptionnet_deepfake.onnx',
            'mobilenet': 'models/deepfake-detection/mobilenet_deepfake.onnx'
        }
        
        if model_name not in model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Check VRAM availability
        vram_requirements = {
            'efficientnet_b4': 44,
            'xceptionnet': 88,
            'mobilenet': 15
        }
        
        required_vram = vram_requirements[model_name]
        if not await self._check_vram_availability(required_vram):
            if model_name != 'mobilenet':
                self.logger.warning(f"Insufficient VRAM for {model_name}, falling back to MobileNet")
                return await self._load_model('mobilenet')
            else:
                raise RuntimeError("Insufficient VRAM for any deepfake detection model")
        
        # Load model with optimizations
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.models[model_name] = ort.InferenceSession(
            model_paths[model_name],
            providers=providers,
            sess_options=session_options
        )
        
        self.current_model = model_name
        
        # Initialize usage statistics
        if model_name not in self.stats['model_usage']:
            self.stats['model_usage'][model_name] = 0
    
    def detect_deepfake(self, face_image: np.ndarray, frame_sequence: List[np.ndarray] = None) -> DeepfakeDetectionResult:
        """
        Detect if face is authentic or deepfake
        
        Args:
            face_image: Face region image
            frame_sequence: Optional sequence for temporal analysis
        
        Returns:
            DeepfakeDetectionResult with detection details
        """
        start_time = time.time()
        
        try:
            # Preprocess face image
            processed_image = self._preprocess_face(face_image)
            
            # Run deepfake detection inference
            model = self.models[self.current_model]
            inputs = {model.get_inputs()[0].name: processed_image}
            outputs = model.run(None, inputs)
            
            # Process model outputs
            authenticity_score, manipulation_probs = self._process_outputs(outputs)
            
            # Determine if face is authentic
            threshold = self.thresholds[self.current_model]['authenticity']
            is_authentic = authenticity_score > threshold
            
            # Identify manipulation type if deepfake
            manipulation_type = self._identify_manipulation_type(manipulation_probs) if not is_authentic else "authentic"
            
            # Calculate confidence
            confidence = self._calculate_confidence(authenticity_score, manipulation_probs)
            
            # Detect visual artifacts
            artifacts = self._detect_visual_artifacts(face_image)
            
            # Perform temporal analysis if sequence provided
            temporal_consistency = None
            if frame_sequence:
                temporal_analysis = self._analyze_temporal_consistency(frame_sequence)
                temporal_consistency = temporal_analysis.frame_consistency
                
                # Adjust results based on temporal analysis
                if temporal_consistency < 0.7 and is_authentic:
                    confidence *= 0.8
                    if confidence < self.thresholds[self.current_model]['confidence']:
                        is_authentic = False
                        manipulation_type = "temporal_manipulation"
            
            # Calculate face quality
            face_quality = self._calculate_face_quality(face_image)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_statistics(is_authentic, manipulation_type, processing_time)
            
            return DeepfakeDetectionResult(
                is_authentic=is_authentic,
                confidence=confidence,
                manipulation_type=manipulation_type,
                authenticity_score=authenticity_score,
                processing_time=processing_time,
                model_used=self.current_model,
                face_quality=face_quality,
                temporal_consistency=temporal_consistency,
                artifacts_detected=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"Deepfake detection failed: {e}")
            raise
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for deepfake detection model"""
        if self.current_model == 'efficientnet_b4':
            # EfficientNet-B4 preprocessing
            face_image = cv2.resize(face_image, (224, 224))
            face_image = face_image.astype(np.float32) / 255.0
            
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            face_image = (face_image - mean) / std
            
        elif self.current_model == 'xceptionnet':
            # Xception preprocessing
            face_image = cv2.resize(face_image, (299, 299))
            face_image = face_image.astype(np.float32) / 255.0
            face_image = (face_image - 0.5) * 2.0  # Scale to [-1, 1]
            
        elif self.current_model == 'mobilenet':
            # MobileNet preprocessing
            face_image = cv2.resize(face_image, (224, 224))
            face_image = face_image.astype(np.float32) / 255.0
            face_image = (face_image - 0.5) * 2.0  # Scale to [-1, 1]
        
        # Add batch dimension and transpose to NCHW format
        face_image = np.transpose(face_image, (2, 0, 1))
        face_image = np.expand_dims(face_image, axis=0)
        
        return face_image
    
    def _process_outputs(self, outputs: List[np.ndarray]) -> Tuple[float, np.ndarray]:
        """Process model outputs to extract authenticity score and manipulation probabilities"""
        if len(outputs) == 1:
            # Binary classification: [fake, real] or [real, fake]
            probs = outputs[0][0]
            if len(probs) == 2:
                authenticity_score = float(probs[1])  # Real class probability
                manipulation_probs = probs
            else:
                # Single output (sigmoid)
                authenticity_score = float(probs[0])
                manipulation_probs = np.array([1 - authenticity_score, authenticity_score])
        else:
            # Multi-output model
            authenticity_score = float(outputs[0][0][1])  # Real class
            manipulation_probs = outputs[1][0] if len(outputs) > 1 else outputs[0][0]
        
        return authenticity_score, manipulation_probs
    
    def _identify_manipulation_type(self, manipulation_probs: np.ndarray) -> str:
        """Identify the type of deepfake manipulation"""
        # This is a simplified version - real implementation would have
        # more sophisticated manipulation type classification
        
        if len(manipulation_probs) > 2:
            # Multi-class model with specific manipulation types
            manipulation_types = [
                "face_swap", "reenactment", "speech_driven", 
                "full_synthesis", "partial"
            ]
            
            max_idx = np.argmax(manipulation_probs[:-1])  # Exclude authentic class
            if max_idx < len(manipulation_types):
                return manipulation_types[max_idx]
        
        # Default classification based on probability patterns
        fake_prob = manipulation_probs[0] if len(manipulation_probs) >= 2 else 1 - manipulation_probs[-1]
        
        if fake_prob > 0.9:
            return "full_synthesis"
        elif fake_prob > 0.7:
            return "face_swap"
        elif fake_prob > 0.5:
            return "partial"
        else:
            return "unknown"
    
    def _calculate_confidence(self, authenticity_score: float, manipulation_probs: np.ndarray) -> float:
        """Calculate confidence score for the detection"""
        # Base confidence on the margin from decision boundary
        margin = abs(authenticity_score - 0.5) * 2
        
        # Adjust based on probability distribution
        if len(manipulation_probs) >= 2:
            prob_spread = np.max(manipulation_probs) - np.min(manipulation_probs)
            confidence = (margin + prob_spread) / 2
        else:
            confidence = margin
        
        return max(0.0, min(1.0, confidence))
    
    def _initialize_artifact_detectors(self) -> Dict[str, callable]:
        """Initialize artifact detection methods"""
        return {
            'compression_artifacts': self._detect_compression_artifacts,
            'blending_boundaries': self._detect_blending_boundaries,
            'inconsistent_lighting': self._detect_lighting_inconsistencies,
            'unnatural_textures': self._detect_texture_anomalies
        }
    
    def _detect_visual_artifacts(self, face_image: np.ndarray) -> List[str]:
        """Detect visual artifacts that may indicate manipulation"""
        artifacts = []
        
        for artifact_type, detector_func in self.artifact_detectors.items():
            try:
                if detector_func(face_image):
                    artifacts.append(artifact_type)
            except Exception as e:
                self.logger.warning(f"Artifact detection failed for {artifact_type}: {e}")
        
        return artifacts
    
    def _detect_compression_artifacts(self, face_image: np.ndarray) -> bool:
        """Detect JPEG compression artifacts"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Apply DCT to detect block artifacts
        height, width = gray.shape
        block_size = 8
        
        artifacts_score = 0
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Check for typical JPEG quantization patterns
                high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
                total_energy = np.sum(np.abs(dct_block))
                
                if total_energy > 0:
                    ratio = high_freq_energy / total_energy
                    if ratio < 0.1:  # Too little high frequency content
                        artifacts_score += 1
        
        total_blocks = ((height // block_size) * (width // block_size))
        return (artifacts_score / total_blocks) > 0.3 if total_blocks > 0 else False
    
    def _detect_blending_boundaries(self, face_image: np.ndarray) -> bool:
        """Detect unnatural blending boundaries"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection to find boundary inconsistencies
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to find connected regions
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Count edge density in different regions
        height, width = gray.shape
        regions = [
            dilated[:height//3, :],           # Top region
            dilated[height//3:2*height//3, :], # Middle region
            dilated[2*height//3:, :]          # Bottom region
        ]
        
        edge_densities = [np.sum(region) / region.size for region in regions]
        
        # Check for unusual edge density variations
        density_std = np.std(edge_densities)
        return density_std > 0.05
    
    def _detect_lighting_inconsistencies(self, face_image: np.ndarray) -> bool:
        """Detect inconsistent lighting patterns"""
        lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Analyze lighting gradients in different face regions
        height, width = l_channel.shape
        
        # Define face regions (simplified)
        regions = {
            'forehead': l_channel[:height//3, width//4:3*width//4],
            'cheeks': l_channel[height//3:2*height//3, :],
            'chin': l_channel[2*height//3:, width//4:3*width//4]
        }
        
        region_means = {name: np.mean(region) for name, region in regions.items()}
        
        # Check for unnatural lighting variations
        lighting_std = np.std(list(region_means.values()))
        return lighting_std > 30  # Threshold for unusual lighting variation
    
    def _detect_texture_anomalies(self, face_image: np.ndarray) -> bool:
        """Detect unnatural texture patterns"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Use Local Binary Patterns for texture analysis
        radius = 3
        n_points = 8 * radius
        
        # Calculate LBP
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                binary_string = ""
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j + radius * np.sin(angle)))
                    
                    if x >= 0 and x < gray.shape[0] and y >= 0 and y < gray.shape[1]:
                        binary_string += "1" if gray[x, y] > center else "0"
                
                lbp[i, j] = int(binary_string, 2) if binary_string else 0
        
        # Calculate texture uniformity
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        uniformity = np.sum(hist ** 2) / (np.sum(hist) ** 2)
        
        # Unnatural textures tend to be either too uniform or too chaotic
        return uniformity < 0.01 or uniformity > 0.05
    
    def _analyze_temporal_consistency(self, frame_sequence: List[np.ndarray]) -> TemporalAnalysis:
        """Analyze temporal consistency across frame sequence"""
        if len(frame_sequence) < 3:
            return TemporalAnalysis(1.0, 1.0, 1.0, 1.0)
        
        # Convert frames to grayscale for analysis
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frame_sequence]
        
        # Frame consistency analysis
        frame_diffs = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            frame_diffs.append(np.mean(diff))
        
        frame_consistency = 1.0 - (np.std(frame_diffs) / np.mean(frame_diffs)) if np.mean(frame_diffs) > 0 else 1.0
        
        # Motion pattern analysis
        motion_vectors = []
        for i in range(1, len(gray_frames)):
            flow = cv2.calcOpticalFlowPyrLK(
                gray_frames[i-1], gray_frames[i],
                np.array([[100, 100]], dtype=np.float32), None
            )[0]
            if flow is not None:
                motion_vectors.append(np.linalg.norm(flow))
        
        motion_patterns = 1.0 - (np.std(motion_vectors) / np.mean(motion_vectors)) if motion_vectors and np.mean(motion_vectors) > 0 else 1.0
        
        # Lighting consistency
        lighting_values = [np.mean(frame) for frame in gray_frames]
        lighting_consistency = 1.0 - (np.std(lighting_values) / np.mean(lighting_values)) if np.mean(lighting_values) > 0 else 1.0
        
        # Compression artifacts (placeholder)
        compression_artifacts = 0.8  # Would implement actual compression analysis
        
        return TemporalAnalysis(
            frame_consistency=max(0.0, min(1.0, frame_consistency)),
            motion_patterns=max(0.0, min(1.0, motion_patterns)),
            lighting_consistency=max(0.0, min(1.0, lighting_consistency)),
            compression_artifacts=compression_artifacts
        )
    
    def _calculate_face_quality(self, face_image: np.ndarray) -> float:
        """Calculate face image quality metrics"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Sharpness using Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 1000, 1.0)
        
        # Brightness and contrast
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        contrast = gray.std()
        contrast_score = min(contrast / 50, 1.0)
        
        # Overall quality
        quality = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        return max(0.0, min(1.0, quality))
    
    def _update_statistics(self, is_authentic: bool, manipulation_type: str, processing_time: float):
        """Update service statistics"""
        self.stats['total_detections'] += 1
        self.stats['model_usage'][self.current_model] += 1
        
        if is_authentic:
            self.stats['authentic_faces'] += 1
        else:
            self.stats['deepfake_faces'] += 1
            if manipulation_type not in self.stats['manipulation_types']:
                self.stats['manipulation_types'][manipulation_type] = 0
            self.stats['manipulation_types'][manipulation_type] += 1
        
        # Update average processing time
        total = self.stats['total_detections']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    async def _check_vram_availability(self, required_mb: int) -> bool:
        """Check if required VRAM is available"""
        # Implementation would query VRAM manager
        return True  # Placeholder
    
    def get_statistics(self) -> dict:
        """Get service performance statistics"""
        total = self.stats['total_detections']
        return {
            'current_model': self.current_model,
            'total_detections': total,
            'authentic_faces': self.stats['authentic_faces'],
            'deepfake_faces': self.stats['deepfake_faces'],
            'deepfake_rate': (
                self.stats['deepfake_faces'] / total if total > 0 else 0
            ),
            'manipulation_types_detected': self.stats['manipulation_types'],
            'average_processing_time': self.stats['average_processing_time'],
            'model_usage': self.stats['model_usage']
        }
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different deepfake detection model"""
        try:
            if model_name not in self.models:
                await self._load_model(model_name)
            else:
                self.current_model = model_name
            
            self.logger.info(f"Switched to deepfake detection model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to model {model_name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        for model in self.models.values():
            del model
        self.models.clear()
```

## API Integration

### REST Endpoints

```python
from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List, Optional
import base64
import io
from PIL import Image

router = APIRouter(prefix="/api/v1/deepfake-detection", tags=["deepfake-detection"])

@router.post("/detect-deepfake")
async def detect_deepfake_image(
    image: UploadFile = File(...),
    model: Optional[str] = None
):
    """Detect if face image is authentic or deepfake"""
    try:
        # Load and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        face_array = np.array(pil_image)
        
        # Switch model if specified
        if model:
            await deepfake_detection_service.switch_model(model)
        
        # Detect deepfake
        result = deepfake_detection_service.detect_deepfake(face_array)
        
        return {
            "success": True,
            "is_authentic": result.is_authentic,
            "confidence": result.confidence,
            "manipulation_type": result.manipulation_type,
            "authenticity_score": result.authenticity_score,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "face_quality": result.face_quality,
            "artifacts_detected": result.artifacts_detected
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-deepfake-sequence")
async def detect_deepfake_video_sequence(
    images: List[UploadFile] = File(...),
    model: Optional[str] = None
):
    """Detect deepfake using video sequence for temporal analysis"""
    try:
        # Load frame sequence
        frame_sequence = []
        for image in images:
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data))
            frame_sequence.append(np.array(pil_image))
        
        if len(frame_sequence) < 1:
            raise HTTPException(
                status_code=400, 
                detail="At least 1 frame required"
            )
        
        # Switch model if specified
        if model:
            await deepfake_detection_service.switch_model(model)
        
        # Use the last frame as primary image
        primary_frame = frame_sequence[-1]
        
        # Detect deepfake with temporal analysis
        result = deepfake_detection_service.detect_deepfake(primary_frame, frame_sequence)
        
        return {
            "success": True,
            "is_authentic": result.is_authentic,
            "confidence": result.confidence,
            "manipulation_type": result.manipulation_type,
            "authenticity_score": result.authenticity_score,
            "temporal_consistency": result.temporal_consistency,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "frame_count": len(frame_sequence),
            "artifacts_detected": result.artifacts_detected
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_deepfake_statistics():
    """Get deepfake detection service statistics"""
    return deepfake_detection_service.get_statistics()

@router.post("/switch-model")
async def switch_deepfake_model(model_name: str):
    """Switch to a different deepfake detection model"""
    success = await deepfake_detection_service.switch_model(model_name)
    
    if success:
        return {"success": True, "current_model": model_name}
    else:
        raise HTTPException(status_code=500, detail="Failed to switch model")
```

## Integration Examples

### Complete Media Authenticity Pipeline

```python
class MediaAuthenticityPipeline:
    def __init__(self, face_detection_service, anti_spoofing_service, deepfake_detection_service):
        self.detection_service = face_detection_service
        self.anti_spoofing_service = anti_spoofing_service
        self.deepfake_service = deepfake_detection_service
    
    async def verify_media_authenticity(self, image: np.ndarray) -> dict:
        """Complete media authenticity verification pipeline"""
        
        # Step 1: Detect faces
        faces = await self.detection_service.detect_faces(image)
        if not faces:
            return {"success": False, "error": "No faces detected"}
        
        results = []
        
        for face in faces:
            face_region = self._extract_face_region(image, face)
            
            # Step 2: Anti-spoofing check
            spoofing_result = self.anti_spoofing_service.detect_spoofing(face_region)
            
            # Step 3: Deepfake detection
            deepfake_result = self.deepfake_service.detect_deepfake(face_region)
            
            # Combine results
            is_authentic = spoofing_result.is_live and deepfake_result.is_authentic
            overall_confidence = min(spoofing_result.confidence, deepfake_result.confidence)
            
            results.append({
                "face_bbox": face.bbox,
                "is_authentic": is_authentic,
                "overall_confidence": overall_confidence,
                "liveness_check": {
                    "is_live": spoofing_result.is_live,
                    "confidence": spoofing_result.confidence,
                    "attack_type": spoofing_result.attack_type
                },
                "deepfake_check": {
                    "is_authentic": deepfake_result.is_authentic,
                    "confidence": deepfake_result.confidence,
                    "manipulation_type": deepfake_result.manipulation_type,
                    "artifacts": deepfake_result.artifacts_detected
                }
            })
        
        # Overall media assessment
        all_authentic = all(result["is_authentic"] for result in results)
        avg_confidence = np.mean([result["overall_confidence"] for result in results])
        
        return {
            "success": True,
            "media_authentic": all_authentic,
            "overall_confidence": avg_confidence,
            "faces_analyzed": len(results),
            "face_results": results
        }
```

## Configuration

### Service Configuration

```yaml
deepfake_detection:
  default_model: "efficientnet_b4"
  
  models:
    efficientnet_b4:
      path: "models/deepfake-detection/efficientnet_b4_deepfake.onnx"
      input_size: [224, 224]
      authenticity_threshold: 0.5
      confidence_threshold: 0.7
      
    xceptionnet:
      path: "models/deepfake-detection/xceptionnet_deepfake.onnx"
      input_size: [299, 299]
      authenticity_threshold: 0.4
      confidence_threshold: 0.75
      
    mobilenet:
      path: "models/deepfake-detection/mobilenet_deepfake.onnx"
      input_size: [224, 224]
      authenticity_threshold: 0.6
      confidence_threshold: 0.65
  
  artifact_detection:
    enable_compression_detection: true
    enable_blending_detection: true
    enable_lighting_analysis: true
    enable_texture_analysis: true
  
  temporal_analysis:
    min_frames_for_analysis: 3
    consistency_threshold: 0.7
    motion_analysis_enable: true
  
  performance:
    max_concurrent_requests: 4
    batch_processing: false
    enable_gpu: true
```

## Best Practices

### 1. Multi-layered Defense
- Combine deepfake detection with anti-spoofing
- Use temporal analysis for video content
- Implement artifact detection as additional verification

### 2. Model Selection Strategy
- Use EfficientNet-B4 for balanced performance
- Switch to XceptionNet for high-stakes verification
- Fallback to MobileNet under VRAM constraints

### 3. Performance Optimization
- Cache preprocessed face regions
- Use appropriate input resolutions for each model
- Implement progressive analysis (fast first, detailed if suspicious)

### 4. Accuracy Considerations
- Validate face quality before analysis
- Consider temporal consistency for video content
- Implement ensemble methods for critical applications

## Troubleshooting

### Common Issues

1. **High False Positives**: Adjust authenticity thresholds, improve preprocessing
2. **Poor Video Analysis**: Check frame quality and temporal resolution
3. **VRAM Issues**: Use model switching or CPU fallback
4. **Artifact Detection Noise**: Fine-tune artifact detection thresholds

### Monitoring

- Track deepfake detection rates and patterns
- Monitor model performance across different content types
- Alert on unusual manipulation type distributions
- Log processing times and resource usage
