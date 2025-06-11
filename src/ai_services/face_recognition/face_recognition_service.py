"""
Enhanced Face Recognition Service with GPU Optimization
เวอร์ชันที่ปรับปรุงแล้วพร้อม GPU optimization สำหรับ RTX 3060
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import cv2
import os

# Conditional import for ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from .models import (
    FaceEmbedding,
    FaceMatch,
    FaceRecognitionResult,
    FaceComparisonResult,
    ModelType
)
from ..common.vram_manager import VRAMManager


@dataclass
class RecognitionConfig:
    """Enhanced Configuration สำหรับ ultra_advanced_test_v14"""
    similarity_threshold: float = 0.60  # ใช้ค่าเดิมจากระบบ
    unknown_threshold: float = 0.55
    max_faces: int = 10
    quality_threshold: float = 0.2
    auto_model_selection: bool = True
    preferred_model: Optional[ModelType] = None
    enable_quality_assessment: bool = True
    enable_unknown_detection: bool = True
    
    # GPU Optimization settings
    batch_size: int = 8
    enable_gpu_optimization: bool = True
    cuda_memory_fraction: float = 0.8
    use_cuda_graphs: bool = True
    parallel_processing: bool = True


class FaceRecognitionService:
    """Enhanced Face Recognition Service with GPU Optimization - เข้ากันได้กับระบบเดิม"""
    
    def __init__(
        self,
        vram_manager: Optional[VRAMManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.vram_manager = vram_manager
        
        # Parse config - ADD MISSING CONFIG ATTRIBUTE
        if config is None:
            config = {}
        self.config = config  # Store config as instance attribute
        self.preferred_model = config.get('preferred_model', 'facenet')
        self.embedding_size = config.get('embedding_size', 512)
        self.threshold = config.get('threshold', 0.6)
        
        # Model management
        self.current_model = None
        self.current_model_type = ModelType.FACENET  # Default
        
        # Face database - เข้ากันได้กับระบบเดิม
        self.face_database: Dict[str, List[FaceEmbedding]] = {}
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'total_extractions': 0,
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'unknown_detections': 0,
            'processing_times': [],
            'batch_processing_count': 0
        }
        
        self.logger.info("🚀 Enhanced Face Recognition Service initialized (GPU Optimized)")

    async def initialize(self) -> bool:
        """Initialize with GPU optimization"""
        try:
            self.logger.info("🔧 Initializing Optimized Face Recognition Service...")
            
            # Load default model with optimization
            success = await self.load_model_optimized(self.current_model_type)
            if not success:
                self.logger.error("❌ Failed to load optimized model")
                return False
            
            # Warm up model
            await self._warmup_model()
            
            self.logger.info("✅ Optimized Face Recognition Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing optimized service: {e}")
            return False

    async def load_model_optimized(self, model_type: ModelType) -> bool:
        """Load model with GPU optimization settings"""
        try:
            if self.current_model_type == model_type and self.current_model is not None:
                return True
            
            # Clean up previous model
            if self.current_model is not None:
                self.current_model = None
              # Get model path
            base_path = os.path.join(os.getcwd(), "model", "face-recognition")
            
            if model_type == ModelType.ADAFACE:
                model_path = os.path.join(base_path, "adaface_ir101.onnx")
            elif model_type == ModelType.ARCFACE:
                model_path = os.path.join(base_path, "arcface_r100.onnx")
            else:  # FaceNet
                model_path = os.path.join(base_path, "facenet_vggface2.onnx")
            
            if not os.path.exists(model_path):
                self.logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            self.logger.info(f"🔄 Loading OPTIMIZED {model_type.value} model from: {model_path}")
            
            # Simplified session options for better compatibility
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            
            # Configure providers with simpler CUDA options
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                cuda_available = False
                
            if cuda_available and self.config.enable_gpu_optimization:
                # Simplified CUDA options for better compatibility
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': int(4.8 * 1024 * 1024 * 1024),  # 4.8GB for RTX 3060
                    'cudnn_conv_algo_search': 'HEURISTIC',
                }
                
                providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
                self.logger.info("🔥 Using SIMPLIFIED CUDA configuration for better compatibility")
            else:
                providers = ['CPUExecutionProvider']
                self.logger.info("💻 Using CPU configuration")
            
            # Create optimized inference session
            self.current_model = ort.InferenceSession(
                model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            self.current_model_type = model_type
            
            # Log optimization details
            if self.current_model is not None:
                input_info = self.current_model.get_inputs()[0]
                output_info = self.current_model.get_outputs()[0]
                self.logger.info("✅ Optimized model loaded successfully:")
                self.logger.info(f"   📊 Input shape: {input_info.shape}")
                self.logger.info(f"   📊 Output shape: {output_info.shape}")
                self.logger.info(f"   🔧 Providers: {self.current_model.get_providers()}")
                self.logger.info(f"   🚀 GPU Optimization: {cuda_available}")
            
            return True
                
        except Exception as e:
            self.logger.error(f"❌ Error loading optimized model {model_type.value}: {e}")
            return False

    async def _warmup_model(self):
        """Warm up model สำหรับ GPU performance"""
        try:
            if self.current_model is None:
                return
            
            self.logger.info("🔥 Warming up GPU model...")
            
            # Create dummy input based on model type
            if self.current_model_type in [ModelType.ADAFACE, ModelType.ARCFACE]:
                dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
            else:  # FaceNet
                dummy_input = np.random.randn(1, 3, 160, 160).astype(np.float32)
            
            # Run warmup iterations
            input_name = self.current_model.get_inputs()[0].name
            
            warmup_start = time.time()
            for i in range(5):
                _ = self.current_model.run(None, {input_name: dummy_input})
            
            warmup_time = time.time() - warmup_start
            self.logger.info(f"🔥 Model warmed up in {warmup_time:.3f}s - Ready for high performance!")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model warmup failed: {e}")

    async def extract_embedding(self, face_image: np.ndarray) -> Optional[FaceEmbedding]:
        """Extract embedding - optimized version but compatible with old interface"""
        try:
            start_time = time.time()
            
            # Ensure model is loaded
            if self.current_model is None:
                if not await self.initialize():
                    return None
            
            # Preprocess image
            input_tensor = self._preprocess_image_optimized(face_image, self.current_model_type)
            if input_tensor is None:
                self.logger.error("❌ Image preprocessing failed")
                return None            # Run optimized inference
            try:
                input_name = self.current_model.get_inputs()[0].name
                
                # Debug input tensor
                self.logger.debug(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
                self.logger.debug(f"Input range: [{np.min(input_tensor):.3f}, {np.max(input_tensor):.3f}]")
                
                outputs = self.current_model.run(None, {input_name: input_tensor})
                
                # Debug outputs
                self.logger.debug(f"Model outputs: {len(outputs) if outputs else 0} outputs")
                if outputs:
                    self.logger.debug(f"Output 0 shape: {outputs[0].shape if len(outputs) > 0 else 'None'}")
                
                # Extract embedding - handle different output formats
                if outputs and len(outputs) > 0:
                    embedding_vector = outputs[0]
                    # Handle batch dimension
                    if len(embedding_vector.shape) > 1:
                        embedding_vector = embedding_vector[0]
                    
                    # Normalize embedding
                    embedding_vector = embedding_vector / (np.linalg.norm(embedding_vector) + 1e-8)
                else:
                    self.logger.error("❌ No outputs received from model")
                    return None
                
            except Exception as model_error:
                self.logger.error(f"❌ Optimized inference failed: {model_error}")
                import traceback
                self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
                return None
            
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
            
            self.logger.debug(f"✅ Optimized embedding extracted: {embedding_vector.shape}, quality={quality_score:.1f}, time={processing_time*1000:.1f}ms")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ Optimized embedding extraction failed: {e}")
            return None

    def _preprocess_image_optimized(self, face_image: np.ndarray, model_type: ModelType) -> Optional[np.ndarray]:
        """Optimized image preprocessing"""
        try:
            # Determine target size
            if model_type in [ModelType.ADAFACE, ModelType.ARCFACE]:
                target_size = (112, 112)
            else:  # FaceNet
                target_size = (160, 160)
            
            # Resize with high quality interpolation
            face_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Convert BGR to RGB
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to 0-1
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Model-specific normalization
            if model_type in [ModelType.ADAFACE, ModelType.ARCFACE]:
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                face_normalized = (face_normalized - mean) / std
            else:  # FaceNet
                face_normalized = (face_normalized - 0.5) * 2.0
            
            # Convert to CHW format and add batch dimension
            input_tensor = np.transpose(face_normalized, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            return input_tensor.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Optimized preprocessing failed: {e}")
            return None

    def _calculate_embedding_quality(self, embedding: np.ndarray) -> float:
        """Calculate embedding quality score (optimized)"""
        try:
            # Vectorized operations for speed
            magnitude = np.linalg.norm(embedding)
            variance = np.var(embedding)
            sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)
            
            # Quality metrics
            magnitude_score = min(float(magnitude), 1.0)
            variance_score = min(float(variance) * 10, 1.0)
            sparsity_score = max(0.0, 1.0 - float(sparsity))
            
            # Combine scores
            quality = (magnitude_score * 0.4 + variance_score * 0.3 + sparsity_score * 0.3)
            return float(np.clip(quality * 100, 0.0, 100.0))
            
        except Exception:
            return 50.0

    # เมธอดที่ต้องมีเพื่อเข้ากันได้กับระบบเดิม
    async def add_face_to_database(self, person_id: str, face_image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add face to database (compatible with old system)"""
        try:
            embedding = await self.extract_embedding(face_image)
            
            if embedding is None:
                self.logger.error(f"❌ Failed to extract optimized embedding for person {person_id}")
                return False
            
            if metadata:
                embedding.metadata = metadata
            
            if person_id not in self.face_database:
                self.face_database[person_id] = []
            
            self.face_database[person_id].append(embedding)
            
            total_embeddings = len(self.face_database[person_id])
            self.logger.info(f"✅ Added optimized face for person {person_id} to database (total embeddings: {total_embeddings})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding face to database: {e}")
            return False

    async def recognize_face(self, face_image: np.ndarray) -> FaceRecognitionResult:
        """Face recognition (compatible with old system interface)"""
        try:
            start_time = time.time()
            
            # Extract optimized embedding
            embedding = await self.extract_embedding(face_image)
            if embedding is None:
                return FaceRecognitionResult(
                    matches=[],
                    best_match=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_used=self.current_model_type
                )
            
            # Search in database with optimized similarity
            matches = []
            for person_id, stored_embeddings in self.face_database.items():
                max_similarity = 0.0
                best_stored_embedding = None
                
                for stored_embedding in stored_embeddings:
                    if stored_embedding.model_type != embedding.model_type:
                        continue
                    
                    if embedding.vector is not None and stored_embedding.vector is not None:
                        similarity = self._cosine_similarity_optimized(
                            embedding.vector, stored_embedding.vector
                        )
                    else:
                        similarity = 0.0
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_stored_embedding = stored_embedding
                
                # Create match if above threshold
                if max_similarity >= self.config.similarity_threshold:
                    match = FaceMatch(
                        person_id=person_id,
                        confidence=max_similarity,
                        embedding=best_stored_embedding
                    )
                    matches.append(match)
            
            # Sort by confidence
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            # Enhanced decision logic
            best_match = None
            confidence = 0.0
            
            if matches:
                top_match = matches[0]
                
                # Quality-based threshold adjustment
                min_confidence_threshold = max(
                    self.config.similarity_threshold,
                    0.6  # Minimum threshold
                )
                
                quality_factor = min(embedding.quality_score / 100.0, 1.0)
                adjusted_threshold = min_confidence_threshold * quality_factor
                
                if top_match.confidence >= adjusted_threshold:
                    best_match = top_match
                    confidence = top_match.confidence
                    self.stats['successful_recognitions'] += 1
                    self.logger.debug(f"✅ Optimized recognition: {top_match.person_id} (confidence: {top_match.confidence:.3f})")
                else:
                    self.stats['unknown_detections'] += 1
                    self.logger.debug(f"🔍 Below threshold: {top_match.person_id} ({top_match.confidence:.3f} < {adjusted_threshold:.3f})")
            else:
                self.stats['unknown_detections'] += 1
                self.logger.debug("🔍 No matches found - Unknown face")
            
            # Update statistics
            self.stats['total_recognitions'] += 1
            processing_time = time.time() - start_time
            
            result = FaceRecognitionResult(
                matches=matches,
                best_match=best_match,
                confidence=confidence,
                processing_time=processing_time,
                model_used=self.current_model_type,
                embedding=embedding.vector if embedding else None
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Optimized face recognition failed: {e}")
            return FaceRecognitionResult(
                matches=[],
                best_match=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=self.current_model_type,
                error=str(e)
            )

    def _cosine_similarity_optimized(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Optimized cosine similarity calculation"""
        try:
            # Vectorized operations for maximum speed
            dot_product = np.dot(embedding1, embedding2)
            norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            
            if norm_product == 0:
                return 0.0
            
            similarity = dot_product / norm_product
            
            # Convert to 0-1 range
            similarity = (similarity + 1.0) / 2.0
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception:
            return 0.0

    async def compare_faces(self, face1: np.ndarray, face2: np.ndarray) -> FaceComparisonResult:
        """Compare faces (compatible interface)"""
        try:
            start_time = time.time()
            
            # Extract embeddings
            embedding1 = await self.extract_embedding(face1)
            embedding2 = await self.extract_embedding(face2)
            
            if embedding1 is None or embedding2 is None:
                return FaceComparisonResult(
                    similarity=0.0,
                    is_same_person=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_used=self.current_model_type
                )
            
            # Calculate optimized similarity
            if embedding1.vector is not None and embedding2.vector is not None:
                similarity = self._cosine_similarity_optimized(embedding1.vector, embedding2.vector)
            else:
                similarity = 0.0
                
            is_same_person = similarity >= self.config.similarity_threshold
            
            result = FaceComparisonResult(
                similarity=similarity,
                is_same_person=is_same_person,
                confidence=similarity,
                processing_time=time.time() - start_time,
                model_used=self.current_model_type
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Optimized face comparison failed: {e}")
            return FaceComparisonResult(
                similarity=0.0,
                is_same_person=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=self.current_model_type,
                error=str(e)
            )

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics (compatible interface)"""
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
            
            # Calculate throughput
            total_time = sum(processing_times) if processing_times else 1
            throughput = self.stats['total_extractions'] / total_time if total_time > 0 else 0
            
            return {
                'current_model': self.current_model_type.value if self.current_model_type else None,
                'model_loaded': self.current_model is not None,
                'database_size': len(self.face_database),
                'total_embeddings': sum(len(embs) for embs in self.face_database.values()),
                'optimization_enabled': self.config.enable_gpu_optimization,
                'performance': {
                    'total_extractions': self.stats['total_extractions'],
                    'total_recognitions': self.stats['total_recognitions'],
                    'successful_recognitions': self.stats['successful_recognitions'],
                    'unknown_detections': self.stats['unknown_detections'],
                    'recognition_rate': recognition_rate,
                    'unknown_detection_rate': unknown_rate,
                    'average_processing_time': avg_time,
                    'average_processing_time_ms': avg_time * 1000,
                    'throughput_fps': throughput,
                    'gpu_providers': self.current_model.get_providers() if self.current_model else []
                },
                'thresholds': {
                    'similarity_threshold': self.config.similarity_threshold,
                    'unknown_threshold': self.config.unknown_threshold,
                    'quality_threshold': self.config.quality_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting performance stats: {e}")
            return {}

    async def cleanup(self):
        """Enhanced cleanup with GPU memory management"""
        try:
            if self.current_model is not None:
                self.current_model = None
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.info("🧹 GPU cache cleared")
            except ImportError:
                pass
            
            self.logger.info("✅ Optimized Face Recognition Service cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")