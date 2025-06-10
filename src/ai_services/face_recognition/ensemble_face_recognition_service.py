"""
Face Recognition Ensemble Service - COMPLETE FIXED VERSION
ระบบจดจำใบหน้าแบบ Ensemble ที่รวม 3 โมเดล: AdaFace, FaceNet, ArcFace
ตามเอกสาร face-recognition-docs.md
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import time
import cv2
import os
import asyncio

# Conditional imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

try:
    from PIL import Image as PIL_Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PIL_Image = None

from .models import (
    FaceEmbedding,
    FaceMatch,
    FaceRecognitionResult,
    FaceComparisonResult,
    RecognitionModel,
    RecognitionQuality
)

# Enable debug logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class EnsembleConfig:
    """การตั้งค่าสำหรับ Ensemble System"""
    # Optimal weights ตามเอกสาร
    adaface_weight: float = 0.25  # AdaFace: 25%
    facenet_weight: float = 0.50  # FaceNet: 50% 
    arcface_weight: float = 0.25  # ArcFace: 25%
    
    # Thresholds สำหรับแต่ละโมเดล
    ensemble_threshold: float = 0.20  # Ensemble threshold
    adaface_threshold: float = 0.20
    facenet_threshold: float = 0.20
    arcface_threshold: float = 0.15
    
    # Performance settings
    enable_gpu_optimization: bool = True
    batch_processing: bool = True
    quality_threshold: float = 0.2
    unknown_threshold: float = 0.55


class FixedSingleModelProcessor:
    """ตัวประมวลผลสำหรับโมเดลเดี่ยว - FIXED version"""
    
    def __init__(self, model_path: str, model_type: str, input_size: Tuple[int, int]):
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.session = None
        self.model_loaded = False
        self.logger = logging.getLogger(f"{__name__}.{model_type}")
        
        # Performance counters
        self.total_count = 0
        self.success_count = 0
        self.total_time = 0.0
        
        # Model-specific preprocessing parameters
        self.preprocessing_config = {
            'adaface': {'mean': 0.5, 'std': 0.5, 'size': (112, 112)},
            'facenet': {'mean': 0.0, 'std': 1.0, 'size': (160, 160)}, 
            'arcface': {'mean': 0.5, 'std': 0.5, 'size': (112, 112)}
        }
    
    async def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล ONNX - FIXED version"""
        try:
            if not ONNX_AVAILABLE:
                self.logger.error("❌ ONNX Runtime not available")
                return False
                
            if not os.path.exists(self.model_path):
                self.logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            self.logger.info(f"🔄 Loading {self.model_type} model...")
            
            # FIXED session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 4
            session_options.intra_op_num_threads = 4
            
            # FIXED providers configuration
            providers = []
            if device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
                self.logger.info(f"🔥 {self.model_type} using CUDA configuration")
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                    'enable_cuda_graph': True
                }))
            providers.append('CPUExecutionProvider')
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            self.model_loaded = True
            device_used = "GPU" if len(providers) > 1 and isinstance(providers[0], tuple) else "CPU"
            self.logger.info(f"✅ {self.model_type} loaded successfully on {device_used}")
            
            # Warmup model for better performance
            await self._warmup_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load {self.model_type}: {e}")
            return False
    
    async def _warmup_model(self) -> None:
        """Warmup model สำหรับ GPU performance"""
        try:
            if not self.model_loaded or self.session is None:
                return
                
            self.logger.info(f"🔥 Warming up {self.model_type} model...")
            config = self.preprocessing_config[self.model_type.lower()]
            
            # Create dummy input
            dummy_input = np.random.randn(1, 3, *config['size']).astype(np.float32)
            input_name = self.session.get_inputs()[0].name
            
            # Run warmup iterations
            warmup_start = time.time()
            for i in range(3):
                result = self.session.run(None, {input_name: dummy_input})
                self.logger.debug(f"🔥 {self.model_type} warmup iteration {i+1} successful")
            
            warmup_time = time.time() - warmup_start
            self.logger.info(f"🔥 {self.model_type} warmed up in {warmup_time:.3f}s")
            
        except Exception as e:
            self.logger.warning(f"⚠️ {self.model_type} warmup failed: {e}")
    
    def preprocess_face(self, face_image) -> Optional[np.ndarray]:
        """เตรียมภาพสำหรับโมเดล - FIXED version"""
        try:
            # FIXED input validation
            if face_image is None:
                self.logger.error(f"❌ Face image is None for {self.model_type}")
                return None
            
            # Convert to numpy array if needed
            if not isinstance(face_image, np.ndarray):
                self.logger.debug(f"🔄 Converting face image from {type(face_image)} to numpy array")
                try:
                    if hasattr(face_image, 'numpy'):
                        face_image = face_image.numpy()
                    elif hasattr(face_image, '__array__'):
                        face_image = np.array(face_image)
                    elif isinstance(face_image, (list, tuple)):
                        face_image = np.array(face_image)
                    else:
                        self.logger.error(f"❌ Cannot convert face image type {type(face_image)}")
                        return None
                except Exception as conv_e:
                    self.logger.error(f"❌ Failed to convert face image: {conv_e}")
                    return None
            
            # Validate shape
            if not hasattr(face_image, 'shape') or len(face_image.shape) < 2:
                self.logger.error(f"❌ Invalid face image shape for {self.model_type}")
                return None
            
            # Ensure contiguous memory layout
            if not face_image.flags['C_CONTIGUOUS']:
                face_image = np.ascontiguousarray(face_image)
            
            # Handle grayscale to RGB conversion
            if len(face_image.shape) == 2:  # Grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif len(face_image.shape) == 3 and face_image.shape[2] == 1:  # Single channel
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            
            # Ensure correct data type
            if face_image.dtype != np.uint8:
                if face_image.dtype in [np.float32, np.float64]:
                    if face_image.max() <= 1.0:
                        face_image = (face_image * 255).astype(np.uint8)
                    else:
                        face_image = np.clip(face_image, 0, 255).astype(np.uint8)
                else:
                    face_image = face_image.astype(np.uint8)
            
            config = self.preprocessing_config[self.model_type.lower()]
            target_size = tuple(config['size'])
            
            # FIXED resize with multiple fallback methods
            face_resized = None
            
            # Method 1: Standard OpenCV resize
            try:
                self.logger.debug(f"🔧 Resizing {self.model_type}: {face_image.shape} -> {target_size}")
                face_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LINEAR)
                self.logger.debug(f"✅ Resize successful: {face_resized.shape}")
            except Exception as resize_e:
                self.logger.debug(f"❌ OpenCV resize method 1 failed: {resize_e}")
                
                # Method 2: Force contiguous array
                try:
                    self.logger.debug("🔄 Trying contiguous array method...")
                    face_copy = np.ascontiguousarray(face_image.copy())
                    face_resized = cv2.resize(face_copy, target_size, interpolation=cv2.INTER_LINEAR)
                    self.logger.debug("✅ Contiguous array method successful")
                except Exception as resize2_e:
                    self.logger.debug(f"❌ Contiguous array method failed: {resize2_e}")
                    
                    # Method 3: PIL fallback if available
                    if PIL_AVAILABLE:
                        try:
                            self.logger.debug("🔄 Trying PIL fallback method...")
                            if len(face_image.shape) == 3:
                                pil_image = PIL_Image.fromarray(face_image)
                                pil_resized = pil_image.resize(target_size, PIL_Image.LANCZOS)
                                face_resized = np.array(pil_resized)
                                self.logger.debug("✅ PIL method successful")
                        except Exception as pil_e:
                            self.logger.debug(f"❌ PIL method failed: {pil_e}")
                    
                    # Method 4: Manual resize (last resort)
                    if face_resized is None:
                        try:
                            self.logger.debug("🔄 Trying manual resize method...")
                            h, w = face_image.shape[:2]
                            target_h, target_w = target_size[1], target_size[0]
                            
                            # Simple nearest neighbor resize
                            indices_h = np.round(np.linspace(0, h-1, target_h)).astype(int)
                            indices_w = np.round(np.linspace(0, w-1, target_w)).astype(int)
                            
                            if len(face_image.shape) == 3:
                                face_resized = face_image[np.ix_(indices_h, indices_w, range(face_image.shape[2]))]
                            else:
                                face_resized = face_image[np.ix_(indices_h, indices_w)]
                            
                            self.logger.debug("✅ Manual resize successful")
                        except Exception as manual_e:
                            self.logger.error(f"❌ Manual resize failed: {manual_e}")
            
            if face_resized is None:
                self.logger.error(f"❌ Failed to resize image for {self.model_type}")
                return None
            
            # Convert BGR to RGB if needed
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Apply model-specific normalization
            if config['mean'] == 0.5:  # AdaFace, ArcFace
                face_normalized = (face_normalized - config['mean']) / config['std']
            # FaceNet uses mean=0, std=1 (no additional normalization needed)
            
            # Transpose to CHW format and add batch dimension
            if len(face_normalized.shape) == 3:
                face_normalized = np.transpose(face_normalized, (2, 0, 1))
            face_normalized = np.expand_dims(face_normalized, axis=0)
            
            self.logger.debug(f"🎯 {self.model_type} preprocessing complete: {face_normalized.shape}")
            
            return face_normalized.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed for {self.model_type}: {e}")
            return None
    
    async def extract_embedding(self, face_image) -> Optional[np.ndarray]:
        """สร้าง embedding จากใบหน้า - FIXED version"""
        try:
            start_time = time.time()
            self.total_count += 1
            
            if not self.model_loaded or self.session is None:
                self.logger.error(f"❌ {self.model_type} model not loaded")
                return None
            
            # FIXED preprocessing
            processed_image = self.preprocess_face(face_image)
            if processed_image is None:
                return None
            
            # FIXED inference
            try:
                input_name = self.session.get_inputs()[0].name
                self.logger.debug(f"🔧 Running inference for {self.model_type}")
                
                # Run inference
                outputs = self.session.run(None, {input_name: processed_image})
                
                if outputs is None or len(outputs) == 0:
                    self.logger.error(f"❌ No outputs returned from {self.model_type} model")
                    return None
                
                embedding_output = outputs[0]
                if embedding_output is None:
                    self.logger.error(f"❌ Null embedding output from {self.model_type}")
                    return None
                
            except Exception as inference_e:
                self.logger.error(f"❌ Inference failed for {self.model_type}: {inference_e}")
                return None
            
            # FIXED embedding processing
            try:
                # Handle different output shapes
                if len(embedding_output.shape) == 0:
                    self.logger.error(f"❌ Scalar embedding output from {self.model_type}")
                    return None
                elif len(embedding_output.shape) == 1:
                    embedding = embedding_output  # Already 1D
                elif len(embedding_output.shape) == 2 and embedding_output.shape[0] == 1:
                    embedding = embedding_output[0]  # Remove batch dimension
                else:
                    self.logger.error(f"❌ Unexpected embedding shape from {self.model_type}: {embedding_output.shape}")
                    return None
                
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 1e-8:
                    embedding = embedding / norm
                else:
                    self.logger.warning(f"⚠️ Very small embedding norm for {self.model_type}: {norm}")
                    return None
                
                self.success_count += 1
                processing_time = time.time() - start_time
                self.total_time += processing_time
                
                self.logger.debug(f"✅ {self.model_type} embedding extracted successfully: shape={embedding.shape}")
                
                return embedding
                
            except Exception as embed_e:
                self.logger.error(f"❌ Embedding processing failed for {self.model_type}: {embed_e}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Embedding extraction failed for {self.model_type}: {e}")
            return None


class EnsembleFaceRecognitionService:
    """Face Recognition Ensemble Service - FIXED VERSION"""
    
    def __init__(self, config: Optional[EnsembleConfig] = None, vram_manager=None):
        self.logger = logging.getLogger(__name__)
        self.config = config or EnsembleConfig()
        self.vram_manager = vram_manager
        
        # FIXED weights
        self.weights = {
            'adaface': self.config.adaface_weight,
            'facenet': self.config.facenet_weight,  
            'arcface': self.config.arcface_weight
        }
        
        # สร้าง processors สำหรับแต่ละโมเดล
        self.models = {
            'adaface': FixedSingleModelProcessor(
                'model/face-recognition/adaface_ir101.onnx',
                'adaface',
                (112, 112)
            ),
            'facenet': FixedSingleModelProcessor(
                'model/face-recognition/facenet_vggface2.onnx', 
                'facenet',
                (160, 160)
            ),
            'arcface': FixedSingleModelProcessor(
                'model/face-recognition/arcface_r100.onnx',
                'arcface', 
                (112, 112)
            )
        }
        
        # Face database
        self.face_database: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance statistics
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'ensemble_processing_times': [],
            'model_performance': {
                'adaface': {'success': 0, 'total': 0, 'avg_time': 0.0},
                'facenet': {'success': 0, 'total': 0, 'avg_time': 0.0},
                'arcface': {'success': 0, 'total': 0, 'avg_time': 0.0}
            }
        }
        
        self.logger.info("🚀 Face Recognition Ensemble Service initialized")
    
    async def initialize(self) -> bool:
        """เริ่มต้นระบบ Ensemble"""
        try:
            self.logger.info("🔧 Initializing Face Recognition Ensemble...")
            
            # โหลดโมเดลทั้งหมดแบบ sequential
            loaded_models = 0
            for model_name, processor in self.models.items():
                success = await processor.load_model("cuda" if self.config.enable_gpu_optimization else "cpu")
                if success:
                    loaded_models += 1
                    self.logger.info(f"✅ {model_name} loaded successfully")
                else:
                    self.logger.error(f"❌ Failed to load {model_name}")
            
            if loaded_models == 0:
                self.logger.error("❌ No models loaded successfully")
                return False
            elif loaded_models < 3:
                self.logger.warning(f"⚠️ Only {loaded_models}/3 models loaded. Ensemble may have reduced accuracy.")
            
            self.logger.info(f"✅ Face Recognition Ensemble initialized with {loaded_models}/3 models")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Ensemble: {e}")
            return False
    
    async def extract_ensemble_embedding(self, face_image) -> Optional[FaceEmbedding]:
        """สร้าง ensemble embedding จากใบหน้า"""
        try:
            start_time = time.time()
            
            # Extract embeddings จากทุกโมเดล
            valid_embeddings = {}
            
            self.logger.debug(f"🔧 Extracting embeddings from {len(self.models)} models")
            
            for model_name, processor in self.models.items():
                if processor.model_loaded:
                    try:
                        embedding = await processor.extract_embedding(face_image)
                        if embedding is not None:
                            valid_embeddings[model_name] = embedding
                            self.logger.debug(f"✅ {model_name} embedding extracted successfully")
                        else:
                            self.logger.warning(f"⚠️ {model_name} returned invalid embedding")
                    except Exception as model_e:
                        self.logger.warning(f"⚠️ {model_name} extraction failed: {model_e}")
            
            if not valid_embeddings:
                self.logger.error("❌ No valid embeddings extracted from any model")
                return None
            
            # สร้าง ensemble embedding ด้วยน้ำหนักที่เหมาะสม
            ensemble_embedding = self._create_ensemble_embedding(valid_embeddings)
            if ensemble_embedding is None:
                return None
            
            processing_time = time.time() - start_time
            self.stats['total_extractions'] += 1
            self.stats['successful_extractions'] += 1
            self.stats['ensemble_processing_times'].append(processing_time)
            
            # ประเมินคุณภาพ
            quality_score = self._assess_ensemble_quality(valid_embeddings, ensemble_embedding)
            
            self.logger.debug(f"✅ Ensemble embedding created using {len(valid_embeddings)}/{len(self.models)} models in {processing_time:.3f}s")
            
            return FaceEmbedding(
                vector=ensemble_embedding,
                model_type=RecognitionModel.ADAFACE,
                quality_score=quality_score,
                extraction_time=processing_time,
                metadata={
                    'ensemble_type': 'weighted_average',
                    'participating_models': list(valid_embeddings.keys()),
                    'model_count': len(valid_embeddings),
                    'weights_used': {k: self.weights[k] for k in valid_embeddings.keys()},
                    'gpu_optimized': self.config.enable_gpu_optimization
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble embedding extraction failed: {e}")
            return None
    
    def _create_ensemble_embedding(self, embeddings: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """สร้าง ensemble embedding ด้วยน้ำหนักที่เหมาะสม"""
        try:
            if not embeddings:
                return None
                
            # คำนวณน้ำหนักที่ปรับให้เฉพาะโมเดลที่มี
            total_weight = sum(self.weights[model] for model in embeddings.keys())
            if total_weight <= 0:
                self.logger.error("❌ Total weight is zero or negative")
                return None
                
            normalized_weights = {model: self.weights[model] / total_weight for model in embeddings.keys()}
            
            # รวม embedding ด้วยน้ำหนัก
            weighted_embeddings = []
            for model_name, embedding in embeddings.items():
                weighted_embedding = embedding * normalized_weights[model_name]
                weighted_embeddings.append(weighted_embedding)
            
            # Sum all weighted embeddings
            ensemble_embedding = np.sum(weighted_embeddings, axis=0)
            
            # Normalize ผลลัพธ์สุดท้าย
            norm = np.linalg.norm(ensemble_embedding)
            if norm > 1e-8:
                ensemble_embedding = ensemble_embedding / norm
            else:
                self.logger.error("❌ Ensemble embedding has zero norm")
                return None
            
            return ensemble_embedding
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble creation failed: {e}")
            # Fallback: ใช้ embedding แรกที่หาได้
            if embeddings:
                return list(embeddings.values())[0]
            return None
    
    def _assess_ensemble_quality(self, individual_embeddings: Dict[str, np.ndarray], 
                                ensemble_embedding: np.ndarray) -> float:
        """ประเมินคุณภาพของ ensemble embedding"""
        try:
            if len(individual_embeddings) < 1:
                return 0.0
            elif len(individual_embeddings) == 1:
                return 0.7  # คุณภาพปานกลางถ้ามีโมเดลเดียว
            
            # ความสอดคล้องระหว่างโมเดล
            embeddings_array = np.array(list(individual_embeddings.values()))
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
            
            # ใช้เฉพาะค่าที่ไม่ใช่ diagonal
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
            similarities = similarity_matrix[mask]
            
            consistency_score = np.mean(similarities) if len(similarities) > 0 else 0.5
            
            # Bonus สำหรับโมเดลครบ
            quality_score = consistency_score
            if len(individual_embeddings) == 3:
                quality_score *= 1.1  # เพิ่ม 10%
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ Quality assessment failed: {e}")
            return 0.6  # คะแนนเริ่มต้น
    
    async def add_face_to_database(self, person_id: str, face_image, 
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """เพิ่มใบหน้าเข้าฐานข้อมูล"""
        try:
            # Extract ensemble embedding
            embedding = await self.extract_ensemble_embedding(face_image)
            if embedding is None:
                self.logger.error(f"❌ Failed to extract embedding for person {person_id}")
                return False
            
            # เพิ่มเข้าฐานข้อมูล
            if person_id not in self.face_database:
                self.face_database[person_id] = []
            
            embedding_data = {
                'embedding': embedding.vector,
                'quality': embedding.quality_score,
                'model_type': embedding.model_type,
                'processing_time': embedding.extraction_time,
                'timestamp': time.time(),
                'metadata': metadata or {},
                'ensemble_info': embedding.metadata
            }
            
            self.face_database[person_id].append(embedding_data)
            
            self.logger.info(f"✅ Added {person_id} to database (Quality: {embedding.quality_score:.3f})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add {person_id} to database: {e}")
            return False
    
    async def recognize_face(self, face_image) -> Optional[FaceRecognitionResult]:
        """จดจำใบหน้า"""
        try:
            start_time = time.time()
            self.stats['total_recognitions'] += 1
            
            # Extract embedding
            query_embedding = await self.extract_ensemble_embedding(face_image)
            if query_embedding is None:
                return None
            
            # หาคนที่ตรงที่สุด
            best_match = None
            best_similarity = -1.0
            
            for person_id, embeddings_list in self.face_database.items():
                for embedding_data in embeddings_list:
                    similarity = np.dot(query_embedding.vector, embedding_data['embedding'])
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = FaceMatch(
                            person_id=person_id,
                            similarity_score=similarity,
                            confidence_score=similarity,
                            model_type=RecognitionModel.ADAFACE,
                            metadata=embedding_data['metadata']
                        )
            
            processing_time = time.time() - start_time
            
            if best_match and best_similarity > self.config.ensemble_threshold:
                self.stats['successful_recognitions'] += 1
                return FaceRecognitionResult(
                    match=best_match,
                    quality=RecognitionQuality.HIGH if best_similarity > 0.8 else RecognitionQuality.MEDIUM,
                    processing_time=processing_time,
                    embedding=query_embedding
                )
            else:
                return FaceRecognitionResult(
                    match=None,
                    quality=RecognitionQuality.LOW,
                    processing_time=processing_time,
                    embedding=query_embedding
                )
            
        except Exception as e:
            self.logger.error(f"❌ Face recognition failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """รับสถิติของระบบ"""
        try:
            stats = self.stats.copy()
            
            # คำนวณ model performance
            for model_name, processor in self.models.items():
                if processor.total_count > 0:
                    stats['model_performance'][model_name].update({
                        'success_rate': processor.success_count / processor.total_count,
                        'avg_time': processor.total_time / processor.total_count,
                        'total_processed': processor.total_count,
                        'successful': processor.success_count
                    })
            
            # คำนวณ average processing time
            if self.stats['ensemble_processing_times']:
                times = self.stats['ensemble_processing_times']
                stats['average_processing_time'] = np.mean(times)
                stats['min_processing_time'] = np.min(times)
                stats['max_processing_time'] = np.max(times)
            else:
                stats['average_processing_time'] = 0.0
                stats['min_processing_time'] = 0.0
                stats['max_processing_time'] = 0.0
            
            # เพิ่มข้อมูลฐานข้อมูล
            stats['database_size'] = len(self.face_database)
            stats['total_embeddings'] = sum(len(embeddings) for embeddings in self.face_database.values())
            
            # คำนวณ success rates
            if stats['total_extractions'] > 0:
                stats['extraction_success_rate'] = stats['successful_extractions'] / stats['total_extractions']
            else:
                stats['extraction_success_rate'] = 0.0
                
            if stats['total_recognitions'] > 0:
                stats['recognition_success_rate'] = stats['successful_recognitions'] / stats['total_recognitions']
            else:
                stats['recognition_success_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {e}")
            return {}
    
    def clear_database(self):
        """ล้างฐานข้อมูลใบหน้า"""
        self.face_database.clear()
        self.logger.info("🗑️ Face database cleared")
    
    def get_database_info(self) -> Dict[str, Any]:
        """รับข้อมูลฐานข้อมูล"""
        return {
            'total_persons': len(self.face_database),
            'total_embeddings': sum(len(embeddings) for embeddings in self.face_database.values()),
            'persons': list(self.face_database.keys())
        }


# Backward compatibility aliases
OptimizedEnsembleFaceRecognitionService = EnsembleFaceRecognitionService
FixedEnsembleFaceRecognitionService = EnsembleFaceRecognitionService
CompletelyFixedEnsembleFaceRecognitionService = EnsembleFaceRecognitionService