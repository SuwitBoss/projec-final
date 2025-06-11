"""
Face Recognition Ensemble Service - COMPLETELY FIXED VERSION
ระบบจดจำใบหน้าแบบ Ensemble ที่รวม 3 โมเดล: AdaFace, FaceNet, ArcFace
แก้ไขปัญหา image preprocessing และ model inference
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import time
import cv2
import os
import asyncio
import platform # Added for OS detection

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
    FaceMatch,
    FaceRecognitionResult,
    RecognitionQuality
)

# Enable debug logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

IS_WINDOWS = platform.system() == "Windows" # Added for OS specific logic

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
    """ตัวประมวลผลสำหรับโมเดลเดี่ยว - COMPLETELY FIXED version"""
    
    def __init__(self, model_path: str, model_type: str, input_size: Tuple[int, int], ensemble_config: EnsembleConfig): # Added ensemble_config
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.config = ensemble_config # Store config
        self.session = None
        self.model_loaded = False
        self.logger = logging.getLogger(f"{__name__}.{model_type}")
        
        # Performance counters
        self.total_count = 0
        self.success_count = 0
        self.total_time = 0.0
        
        # Model-specific preprocessing parameters
        self.preprocessing_config = {
            'adaface': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'size': (112, 112)},
            'facenet': {'mean': [127.5, 127.5, 127.5], 'std': [128.0, 128.0, 128.0], 'size': (160, 160)}, 
            'arcface': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'size': (112, 112)}
        }
    
    async def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล ONNX - COMPLETELY FIXED version - Optimized"""
        try:
            if not ONNX_AVAILABLE:
                self.logger.error("❌ ONNX Runtime not available")
                return False
                
            if not os.path.exists(self.model_path):
                self.logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            self.logger.info(f"🔄 Loading {self.model_type} model (Optimized)...")
            
            session_options = ort.SessionOptions()
            # Use ORT_ENABLE_EXTENDED for more optimizations
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # Keep this for stability
            session_options.inter_op_num_threads = 2 # Keep this or adjust based on CPU
            session_options.intra_op_num_threads = 2 # Keep this or adjust based on CPU
            
            providers = []
            
            if device == "cuda" and self.config.enable_gpu_optimization: # Check global config
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.logger.info(f"🔥 {self.model_type} using OPTIMIZED CUDA configuration")
                        # Simplified and potentially more robust CUDA options from face_recognition_service.py
                        cuda_options = {
                            'device_id': 0,
                            'arena_extend_strategy': 'kSameAsRequested',
                            # Adjusted GPU memory limit, ensure this is suitable for ensemble
                            'gpu_mem_limit': int(2 * 1024 * 1024 * 1024), # 2GB per model, adjust as needed
                            'cudnn_conv_algo_search': 'HEURISTIC',
                            # 'do_copy_in_default_stream': True, # Removed for simplicity, like in face_recognition_service
                        }
                        providers.append(('CUDAExecutionProvider', cuda_options))
                        self.logger.info(f"🔥 {self.model_type} configured with CUDA options: {cuda_options}")
                    else:
                        self.logger.warning(f"⚠️ CUDA not available according to torch, {self.model_type} using CPU.")
                except ImportError:
                    self.logger.warning(f"⚠️ PyTorch not found, cannot confirm CUDA availability for {self.model_type}. Assuming CPU.")
                except Exception as e:
                    self.logger.warning(f"⚠️ CUDA setup failed for {self.model_type}, using CPU. Error: {e}")
            
            providers.append('CPUExecutionProvider') # Always have CPU as a fallback

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            self.model_loaded = True
            actual_providers = self.session.get_providers()
            device_used = "GPU" if any('CUDAExecutionProvider' in p for p in actual_providers) else "CPU"
            self.logger.info(f"✅ {self.model_type} loaded successfully on {device_used} with providers: {actual_providers}")
            
            await self._warmup_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load {self.model_type}: {e}", exc_info=True)
            self.model_loaded = False
            return False
    
    async def _warmup_model(self) -> None:
        """Warmup model สำหรับ GPU performance - Optimized"""
        try:
            if not self.model_loaded or self.session is None:
                return
                
            self.logger.info(f"🔥 Warming up {self.model_type} model (5 iterations)...")
            config = self.preprocessing_config[self.model_type.lower()]
            
            dummy_input_shape = (1, 3, config['size'][1], config['size'][0]) # NCHW, size is (width, height)
            dummy_input = np.random.randn(*dummy_input_shape).astype(np.float32)
            input_name = self.session.get_inputs()[0].name
            
            warmup_start = time.time()
            for i in range(5): # Increased iterations to 5
                try:
                    _ = self.session.run(None, {input_name: dummy_input})
                    self.logger.debug(f"🔥 {self.model_type} warmup iteration {i+1} successful")
                except Exception as warmup_error:
                    self.logger.warning(f"⚠️ {self.model_type} warmup iteration {i+1} failed: {warmup_error}")
            
            warmup_time = time.time() - warmup_start
            self.logger.info(f"🔥 {self.model_type} warmed up in {warmup_time:.3f}s")
            
        except Exception as e:
            self.logger.warning(f"⚠️ {self.model_type} warmup failed: {e}")
    
    def preprocess_face(self, face_image) -> Optional[np.ndarray]:
        """เตรียมภาพสำหรับโมเดล - COMPLETELY FIXED version - Optimized"""
        try:
            # STEP 1: Input validation and conversion (largely unchanged)
            if face_image is None:
                self.logger.error(f"❌ Face image is None for {self.model_type}")
                return None
            
            # Convert to numpy array if needed
            if not isinstance(face_image, np.ndarray):
                # ... (conversion logic as before)
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

            if not hasattr(face_image, 'shape') or len(face_image.shape) < 2:
                self.logger.error(f"❌ Invalid face image shape for {self.model_type}: {getattr(face_image, 'shape', 'no shape')}")
                return None
            
            if not face_image.flags['C_CONTIGUOUS']:
                face_image = np.ascontiguousarray(face_image)
            
            # STEP 2: Color space handling
            
            processed_cv_image = face_image
            if len(processed_cv_image.shape) == 2:
                processed_cv_image = cv2.cvtColor(processed_cv_image, cv2.COLOR_GRAY2BGR)
            elif len(processed_cv_image.shape) == 3 and processed_cv_image.shape[2] == 1:
                processed_cv_image = cv2.cvtColor(processed_cv_image, cv2.COLOR_GRAY2BGR)
            elif len(processed_cv_image.shape) == 3 and processed_cv_image.shape[2] == 4: # RGBA or BGRA
                # Assuming input might be BGRA from some sources, convert to BGR
                processed_cv_image = cv2.cvtColor(processed_cv_image, cv2.COLOR_BGRA2BGR)


            # Explicit BGR to RGB conversion (models usually expect RGB)
            # This should happen after ensuring it's 3-channel BGR
            if len(processed_cv_image.shape) == 3 and processed_cv_image.shape[2] == 3:
                 processed_cv_image = cv2.cvtColor(processed_cv_image, cv2.COLOR_BGR2RGB)
            else:
                self.logger.error(f"❌ Image is not 3-channel BGR after initial conversion: {processed_cv_image.shape}")
                return None

            # STEP 3: Data type handling (largely unchanged)
            if processed_cv_image.dtype != np.uint8:
                if processed_cv_image.dtype in [np.float32, np.float64]:
                    if processed_cv_image.max() <= 1.0 and processed_cv_image.min() >=0.0 :
                        processed_cv_image = (processed_cv_image * 255).astype(np.uint8)
                    else:
                        processed_cv_image = np.clip(processed_cv_image, 0, 255).astype(np.uint8)
                else:
                    try:
                        processed_cv_image = processed_cv_image.astype(np.uint8)
                    except Exception as dtype_e:
                        self.logger.error(f"❌ Failed to convert face image to uint8: {dtype_e}")
                        return None

            # STEP 4: Resize with INTER_LANCZOS4
            config = self.preprocessing_config[self.model_type.lower()]
            target_size = tuple(config['size']) # (width, height)
            
            face_resized = None
            # self.logger.debug(f"🔧 Resizing {self.model_type}: {processed_cv_image.shape} -> {target_size}")
            
            try:
                # Use INTER_LANCZOS4 for higher quality resize
                face_resized = cv2.resize(processed_cv_image, target_size, interpolation=cv2.INTER_LANCZOS4)
                # self.logger.debug(f"✅ OpenCV resize successful (LANCZOS4): {face_resized.shape}")
            except Exception as resize_e:
                self.logger.warning(f"❌ OpenCV LANCZOS4 resize failed: {resize_e}. Trying INTER_LINEAR...")
                try:
                    face_resized = cv2.resize(processed_cv_image, target_size, interpolation=cv2.INTER_LINEAR)
                    # self.logger.debug(f"✅ OpenCV resize successful (LINEAR): {face_resized.shape}")
                except Exception as linear_resize_e:
                    self.logger.error(f"❌ OpenCV LINEAR resize also failed: {linear_resize_e}")
                    # PIL fallback (as before, if needed and available)
                    if PIL_AVAILABLE:
                        try:
                            self.logger.debug("🔄 Trying PIL resize...")
                            pil_image = PIL_Image.fromarray(processed_cv_image) # Assumes RGB now
                            pil_resized = pil_image.resize(target_size, PIL_Image.LANCZOS)
                            face_resized = np.array(pil_resized)
                            self.logger.debug("✅ PIL resize successful")
                        except Exception as pil_e:
                            self.logger.error(f"❌ PIL resize failed: {pil_e}")
            
            if face_resized is None:
                self.logger.error(f"❌ All resize methods failed for {self.model_type}")
                return None
            
            # STEP 5: Normalization (logic remains similar, applied to RGB image)
            face_normalized = face_resized.astype(np.float32)
            
            if self.model_type.lower() == 'facenet':
                face_normalized = (face_normalized - 127.5) / 128.0
            else: # AdaFace, ArcFace
                face_normalized = face_normalized / 255.0
                face_normalized = (face_normalized - 0.5) / 0.5
            
            # STEP 6: Transpose and add batch dimension (unchanged)
            if len(face_normalized.shape) == 3:
                face_normalized = np.transpose(face_normalized, (2, 0, 1)) # HWC to CHW
            face_normalized = np.expand_dims(face_normalized, axis=0)
            
            expected_shape = (1, 3, target_size[1], target_size[0]) 
            if face_normalized.shape != expected_shape:
                self.logger.error(f"❌ Final shape mismatch for {self.model_type}: {face_normalized.shape} != {expected_shape}")
                return None 
            
            # self.logger.debug(f"🎯 {self.model_type} preprocessing complete: {face_normalized.shape}")
            return face_normalized.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed for {self.model_type}: {e}", exc_info=True)
            return None
    
    async def extract_embedding(self, face_image) -> Optional[np.ndarray]:
        """สร้าง embedding จากใบหน้า - COMPLETELY FIXED version"""
        try:
            start_time = time.time()
            self.total_count += 1
            
            if not self.model_loaded or self.session is None:
                self.logger.error(f"❌ {self.model_type} model not loaded")
                return None
            
            processed_image = self.preprocess_face(face_image)
            if processed_image is None:
                self.logger.error(f"❌ Preprocessing failed for {self.model_type}")
                return None
            
            outputs = None
            try:
                input_name = self.session.get_inputs()[0].name
                self.logger.debug(f"🔧 Running inference for {self.model_type} with input shape {processed_image.shape}")
                
                if not isinstance(processed_image, np.ndarray):
                    self.logger.error(f"❌ Processed image is not numpy array: {type(processed_image)}")
                    return None
                
                if processed_image.size == 0:
                    self.logger.error("❌ Processed image is empty")
                    return None

                # Timeout handling for inference
                if not IS_WINDOWS:
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Model inference timeout for {self.model_type}")

                    original_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(10)  # 10 seconds timeout
                    try:
                        outputs = self.session.run(None, {input_name: processed_image})
                    finally:
                        signal.alarm(0)  # Cancel timeout
                        signal.signal(signal.SIGALRM, original_handler) # Restore original handler
                else:
                    # On Windows, run without signal-based timeout
                    # For true timeout, would need multiprocessing or threading with wait
                    self.logger.debug(f"ℹ️ Running {self.model_type} inference on Windows without signal-based timeout.")
                    outputs = self.session.run(None, {input_name: processed_image})
                
                if outputs is None or len(outputs) == 0:
                    self.logger.error(f"❌ No outputs returned from {self.model_type} model")
                    return None
                
                embedding_output = outputs[0]
                if embedding_output is None:
                    self.logger.error(f"❌ Null embedding output from {self.model_type}")
                    return None
                
            except TimeoutError as te:
                self.logger.error(f"❌ {self.model_type} inference timeout: {te}")
                return None
            except Exception as inference_e:
                self.logger.error(f"❌ Inference failed for {self.model_type}: {inference_e}", exc_info=True)
                return None
            
            try:
                if isinstance(embedding_output, (list, tuple)):
                    embedding_output = embedding_output[0] if len(embedding_output) > 0 else None
                
                if embedding_output is None:
                    self.logger.error(f"❌ Empty embedding output from {self.model_type}")
                    return None
                
                if not isinstance(embedding_output, np.ndarray):
                    embedding_output = np.array(embedding_output)
                
                if embedding_output.size == 0:
                    self.logger.error(f"❌ Zero-size embedding from {self.model_type}")
                    return None
                
                # Standardize embedding shape to 1D
                embedding = embedding_output.flatten()
                
                if embedding.size == 0: # Should not happen if previous checks passed
                    self.logger.error(f"❌ Empty embedding after processing for {self.model_type}")
                    return None
                
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    self.logger.error(f"❌ Invalid values (NaN/Inf) in embedding for {self.model_type}")
                    # Optionally, try to clean or return None. Returning None is safer.
                    return None 
                
                # Normalize embedding (L2 normalization)
                norm = np.linalg.norm(embedding)
                if norm > 1e-8: # Avoid division by zero or very small norm
                    embedding = embedding / norm
                else:
                    self.logger.warning(f"⚠️ Very small embedding norm for {self.model_type}: {norm}. Embedding might be unreliable.")
                    # Depending on policy, could return None or the (almost) zero vector.
                    # For now, let it pass but it's a sign of a problem.
                    # To be stricter: return None

                self.success_count += 1
                processing_time = time.time() - start_time
                self.total_time += processing_time
                
                self.logger.debug(f"✅ {self.model_type} embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f}, time={processing_time:.4f}s")
                
                return embedding
                
            except Exception as embed_e:
                self.logger.error(f"❌ Embedding processing failed for {self.model_type}: {embed_e}", exc_info=True)
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Embedding extraction failed for {self.model_type}: {e}", exc_info=True)
            return None


class EnsembleFaceRecognitionService:
    """Face Recognition Ensemble Service - COMPLETELY FIXED VERSION"""
    
    def __init__(self, config: Optional[EnsembleConfig] = None, vram_manager=None): # vram_manager not used yet
        self.logger = logging.getLogger(__name__)
        self.config = config or EnsembleConfig() # Ensure config is used by FixedSingleModelProcessor
        self.vram_manager = vram_manager 
        
        self.weights = {
            'adaface': self.config.adaface_weight,
            'facenet': self.config.facenet_weight,  
            'arcface': self.config.arcface_weight
        }
        
        self.models = {
            'adaface': FixedSingleModelProcessor(
                'model/face-recognition/adaface_ir101.onnx', 
                'adaface',
                (112, 112),
                self.config # Pass ensemble config
            ),
            'facenet': FixedSingleModelProcessor(
                'model/face-recognition/facenet_vggface2.onnx', 
                'facenet',
                (160, 160),
                self.config # Pass ensemble config
            ),
            'arcface': FixedSingleModelProcessor(
                'model/face-recognition/arcface_r100.onnx',
                'arcface', 
                (112, 112),
                self.config # Pass ensemble config
            )
        }
        
        self.face_database: Dict[str, List[Dict[str, Any]]] = {}
        
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'ensemble_processing_times': [],
            'model_performance': { # This structure might be redundant if FixedSingleModelProcessor tracks its own stats
                'adaface': {'success': 0, 'total': 0, 'avg_time': 0.0},
                'facenet': {'success': 0, 'total': 0, 'avg_time': 0.0},
                'arcface': {'success': 0, 'total': 0, 'avg_time': 0.0}
            }
        }
        
        self.logger.info("🚀 Face Recognition Ensemble Service initialized")
    
    async def initialize(self) -> bool:
        """เริ่มต้นระบบ Ensemble"""
        self.logger.info("🔧 Initializing EnsembleFaceRecognitionService...")
        
        # Load models concurrently
        load_tasks = [model.load_model(device="cuda" if self.config.enable_gpu_optimization else "cpu") 
                      for model in self.models.values()]
        results = await asyncio.gather(*load_tasks)
        
        if not all(results):
            self.logger.error("❌ Failed to initialize one or more models in the ensemble.")
            return False
            
        self.logger.info("✅ EnsembleFaceRecognitionService initialized successfully.")
        return True

    async def extract_ensemble_embedding(self, face_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """สร้าง embedding แบบ ensemble จากใบหน้า"""
        self.stats['total_extractions'] += 1
        start_time = time.time()
        
        # Extract embeddings from all models concurrently
        extraction_tasks = {
            name: model.extract_embedding(face_image)
            for name, model in self.models.items()
        }
        
        # Use asyncio.gather to run extractions concurrently
        # Store results in a dictionary to maintain model-embedding association
        embedding_results = {}
        results_list = await asyncio.gather(*(extraction_tasks.values()))
        
        for i, model_name in enumerate(extraction_tasks.keys()):
            embedding_results[model_name] = results_list[i]

        # Filter out failed extractions
        successful_embeddings = {
            name: emb for name, emb in embedding_results.items() if emb is not None
        }
        
        if not successful_embeddings:
            self.logger.warning("⚠️ All models failed to extract embedding.")
            return None
            
        # Create ensemble embedding
        ensemble_embedding, quality_score = self._create_ensemble_embedding(successful_embeddings)
        
        if ensemble_embedding is None:
            self.logger.warning("⚠️ Failed to create ensemble embedding.")
            return None
            
        self.stats['successful_extractions'] += 1
        processing_time = time.time() - start_time
        self.stats['ensemble_processing_times'].append(processing_time)
        
        return {
            'embedding': ensemble_embedding,
            'quality': quality_score,
            'individual_embeddings': successful_embeddings, # For analysis
            'processing_time': processing_time
        }

    def _create_ensemble_embedding(self, embeddings: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
        """สร้าง embedding แบบ ensemble จาก embeddings ของแต่ละโมเดล"""
        if not embeddings:
            return None, 0.0

        # Weighted average of embeddings
        # Ensure all embeddings are 1D arrays of the same length (or handle appropriately)
        # For simplicity, assume all models output embeddings that can be meaningfully combined.
        # This might require padding or truncation if lengths differ, but ONNX models usually have fixed output sizes.
        
        # Check for consistent embedding sizes (optional, but good practice)
        # For now, assume they are compatible or that the first one sets the standard.
        
        # Use a list to store valid embeddings for averaging
        valid_embeddings = []
        total_weight = 0.0
        
        # Collect valid embeddings and their weights
        for model_name, emb in embeddings.items():
            if emb is not None and model_name in self.weights:
                # Ensure embedding is 1D
                if emb.ndim > 1:
                    emb = emb.flatten() # Or handle error
                
                valid_embeddings.append(emb * self.weights[model_name])
                total_weight += self.weights[model_name]
            else:
                self.logger.warning(f"⚠️ Skipping {model_name} embedding (None or no weight).")

        if not valid_embeddings or total_weight == 0:
            self.logger.error("❌ No valid embeddings to create ensemble or total_weight is zero.")
            return None, 0.0

        # Sum weighted embeddings
        # Need to ensure all embeddings in valid_embeddings have the same shape for direct sum
        # A more robust approach would be to check shapes or resize/pad, but let's assume compatibility for now.
        try:
            # Check if all embeddings have the same shape
            first_shape = valid_embeddings[0].shape
            if not all(emb.shape == first_shape for emb in valid_embeddings):
                self.logger.error(f"❌ Inconsistent embedding shapes for ensemble. First shape: {first_shape}. All shapes: {[e.shape for e in valid_embeddings]}")
                # Fallback: Use only the embedding from the highest weighted model if shapes differ
                # This is a simple fallback; more sophisticated strategies could be used.
                if 'facenet' in embeddings and embeddings['facenet'] is not None: # Assuming FaceNet has highest weight
                    self.logger.warning("⚠️ Fallback: Using FaceNet embedding due to shape mismatch.")
                    fallback_emb = embeddings['facenet'].flatten()
                    norm = np.linalg.norm(fallback_emb)
                    if norm > 1e-8: 
                        fallback_emb = fallback_emb / norm
                    return fallback_emb, self._assess_ensemble_quality(embeddings) # Recalculate quality
                else: # If even FaceNet is not available or fails
                    self.logger.error("❌ Fallback failed: FaceNet embedding not available.")
                    return None, 0.0

            ensemble_embedding_sum = np.sum(valid_embeddings, axis=0)
            
            # Normalize by total weight
            final_ensemble_embedding = ensemble_embedding_sum / total_weight
            
            # L2 normalize the final ensemble embedding
            norm = np.linalg.norm(final_ensemble_embedding)
            if norm > 1e-8:
                final_ensemble_embedding /= norm
            else:
                self.logger.warning("⚠️ Ensemble embedding has very small norm. May be unreliable.")
                # Could return None here if this is considered a failure.
            
            quality_score = self._assess_ensemble_quality(embeddings) # Assess quality based on original embeddings
            
            return final_ensemble_embedding, quality_score
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble embedding: {e}", exc_info=True)
            # Fallback strategy: if weighted sum fails, try to use the embedding from the highest-weighted model
            if 'facenet' in embeddings and embeddings['facenet'] is not None: # Assuming FaceNet has highest weight
                self.logger.warning("⚠️ Fallback: Using FaceNet embedding due to error in weighted sum.")
                fallback_emb = embeddings['facenet'].flatten()
                norm = np.linalg.norm(fallback_emb)
                if norm > 1e-8: 
                    fallback_emb = fallback_emb / norm
                return fallback_emb, self._assess_ensemble_quality(embeddings)
            else:
                self.logger.error("❌ Fallback failed: FaceNet embedding not available after sum error.")
                return None, 0.0

    def _assess_ensemble_quality(self, embeddings: Dict[str, np.ndarray]) -> float:
        """ประเมินคุณภาพของ ensemble embedding (ความสอดคล้องกันของ individual embeddings)"""
        if len(embeddings) < 2:
            return 0.5  # Not enough embeddings for comparison, default to medium quality

        # Calculate pairwise cosine similarity between all available embeddings
        # This gives an idea of how consistent the models are.
        embedding_list = [emb for emb in embeddings.values() if emb is not None]
        if len(embedding_list) < 2:
            return 0.5 # Still not enough after filtering Nones

        # Normalize all embeddings before comparison (should already be done by extract_embedding)
        # For safety, re-normalize here or ensure it's consistently done.
        # normalized_embeddings = []
        # for emb in embedding_list:
        #     norm = np.linalg.norm(emb)
        #     if norm > 1e-8:
        #         normalized_embeddings.append(emb / norm)
        #     else: # Handle zero/small norm embeddings if they weren't filtered
        #         # normalized_embeddings.append(emb) # Or skip
        #         pass # Skip if norm is too small, as it indicates an issue

        # if len(normalized_embeddings) < 2: return 0.5

        # Use already normalized embeddings from extract_embedding
        
        similarities = []
        for i in range(len(embedding_list)):
            for j in range(i + 1, len(embedding_list)):
                # Cosine similarity: dot(A, B) / (norm(A) * norm(B))
                # Since embeddings are L2 normalized, norm(A) = norm(B) = 1
                # So, similarity = dot(A, B)
                sim = np.dot(embedding_list[i], embedding_list[j])
                similarities.append(sim)
        
        if not similarities:
            return 0.0 # Should not happen if len(embedding_list) >= 2

        avg_similarity = np.mean(similarities)
        
        # Scale similarity (usually -1 to 1, but for face embeddings, typically 0 to 1)
        # to a quality score (0 to 1)
        # Example scaling: (avg_similarity + 1) / 2 if range is -1 to 1
        # If embeddings are from similar tasks, dot product of normalized vectors is often positive.        # Let's assume similarity is already in a good range (e.g., 0 to 1 for positive correlations)
        # A simple quality score could be the average similarity itself, if it's well-behaved.
        # Or, map it: e.g., if avg_similarity < 0.3 -> low, 0.3-0.6 -> medium, >0.6 -> high
        
        quality_score = max(0.0, min(1.0, avg_similarity)) # Clamp to [0,1]
        
        self.logger.debug(f"📊 Ensemble quality assessment: avg_similarity={avg_similarity:.4f}, quality_score={quality_score:.4f}")
        return quality_score
        
    async def add_face_to_database(self, person_id: str, face_image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """เพิ่มใบหน้าเข้าฐานข้อมูล"""
        ensemble_result = await self.extract_ensemble_embedding(face_image)
        
        if ensemble_result is None or ensemble_result['embedding'] is None:
            self.logger.error(f"❌ Failed to extract ensemble embedding for {person_id}")
            return False
            
        embedding_data = {
            'embedding': ensemble_result['embedding'],
            'quality': ensemble_result['quality'],
            'model_source': 'ensemble',
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        if person_id not in self.face_database:
            self.face_database[person_id] = []
        self.face_database[person_id].append(embedding_data)
        
        self.logger.info(f"👤 Added face for {person_id} to database. Quality: {ensemble_result['quality']:.3f}")
        return True
        
    async def recognize_face(self, face_image: np.ndarray, top_n: int = 1) -> Optional[FaceRecognitionResult]:
        """จดจำใบหน้าจากภาพ"""
        self.stats['total_recognitions'] += 1
        start_time = time.time()
        
        ensemble_result = await self.extract_ensemble_embedding(face_image)
        
        if ensemble_result is None or ensemble_result['embedding'] is None:
            self.logger.warning("⚠️ Recognition failed: Could not extract ensemble embedding.")
            return FaceRecognitionResult(
                matches=[],
                best_match=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                error="Failed to extract ensemble embedding.",
                embedding_quality=RecognitionQuality.UNKNOWN
            )
            
        query_embedding = ensemble_result['embedding']
        quality_score = ensemble_result['quality']
        
        if quality_score < self.config.quality_threshold:
            self.logger.warning(f"⚠️ Low quality ensemble embedding ({quality_score:.3f} < {self.config.quality_threshold}). Recognition might be unreliable.")
            # Proceed with recognition but flag low quality.
            # Or, could return early if policy dictates.

        matches: List[FaceMatch] = []
        
        for person_id, embeddings_data in self.face_database.items():
            best_person_similarity = -1.0
            
            for db_entry in embeddings_data:
                db_embedding = db_entry['embedding']
                # Cosine similarity for L2 normalized embeddings
                similarity = np.dot(query_embedding, db_embedding)
                if similarity > best_person_similarity:
                    best_person_similarity = similarity

            if best_person_similarity > -1.0: # If any embedding was compared for this person
                  if similarity > best_person_similarity:
                    best_person_similarity = similarity
            
            if best_person_similarity > -1.0: # If any embedding was compared for this person
                matches.append(FaceMatch(
                    person_id=person_id,
                    confidence=float(best_person_similarity), # Ensure float
                    embedding=None # We don't need to pass the embedding here
                ))
                
        # Sort matches by confidence (descending)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        best_match: Optional[FaceMatch] = None
        final_confidence = 0.0
        
        if matches:
            # Apply ensemble threshold
            if matches[0].confidence >= self.config.ensemble_threshold:
                best_match = matches[0]
                final_confidence = matches[0].confidence
                self.stats['successful_recognitions'] += 1
                self.logger.info(f"✅ Recognized: {best_match.person_id} with confidence {best_match.confidence:.4f}")
            else:
                self.logger.info(f"ℹ️ No match above ensemble threshold {self.config.ensemble_threshold}. Best was {matches[0].person_id} at {matches[0].confidence:.4f}")
                # If below unknown_threshold, consider it unknown
                if matches[0].confidence < self.config.unknown_threshold: # Check against unknown_threshold
                    self.logger.info(f"ℹ️ Best match confidence {matches[0].confidence:.4f} is below unknown_threshold {self.config.unknown_threshold}. Considered Unknown.")
                    # best_match remains None, or set to an "Unknown" FaceMatch if desired by API
                else:
                    # Between ensemble_threshold and unknown_threshold - low confidence match
                    # Depending on policy, this could still be a tentative match or unknown.
                    # For now, if it didn't pass ensemble_threshold, it's not a confident match.
                    best_match = None # Explicitly not a confident match
                    final_confidence = matches[0].confidence # Report the highest confidence found

        else: # No entries in database or no matches at all
            self.logger.info("ℹ️ No matches found (database might be empty or no similarity calculated).")

        processing_time = time.time() - start_time
          # Using embedding_quality to store quality_score as per FaceRecognitionResult definition
        # Convert quality_score to RecognitionQuality enum
        quality_enum = RecognitionQuality.UNKNOWN
        if quality_score >= 0.8:
            quality_enum = RecognitionQuality.HIGH
        elif quality_score >= 0.5:
            quality_enum = RecognitionQuality.MEDIUM
        elif quality_score > 0:
            quality_enum = RecognitionQuality.LOW
            
        return FaceRecognitionResult(
            matches=matches[:top_n], # Return top_n matches
            best_match=best_match,
            confidence=float(final_confidence), # Ensure float
            processing_time=processing_time,
            embedding_quality=quality_enum # Use the proper field for quality
        )

    def get_statistics(self) -> Dict[str, Any]:
        """ดึงสถิติการทำงานของ Ensemble System"""
        # Consolidate stats from individual model processors
        model_perf = {}
        for name, model_proc in self.models.items():
            avg_time = (model_proc.total_time / model_proc.total_count) if model_proc.total_count > 0 else 0
            success_rate = (model_proc.success_count / model_proc.total_count) if model_proc.total_count > 0 else 0
            model_perf[name] = {
                'total_extractions': model_proc.total_count,
                'successful_extractions': model_proc.success_count,
                'average_extraction_time_ms': avg_time * 1000,
                'success_rate': success_rate
            }
            
        # Update self.stats with the latest from individual models
        # This overwrites the placeholder structure in self.stats['model_performance']
        self.stats['model_performance'] = model_perf 
        
        # Calculate average ensemble processing time
        if self.stats['ensemble_processing_times']:
            avg_ensemble_time_ms = np.mean(self.stats['ensemble_processing_times']) * 1000
        else:
            avg_ensemble_time_ms = 0
            
        self.stats['average_ensemble_processing_time_ms'] = avg_ensemble_time_ms
        
        return self.stats

    def clear_database(self) -> None:
        """ล้างฐานข้อมูลใบหน้า"""
        self.face_database.clear()
        self.logger.info("🗑️ Face database cleared.")

    def get_database_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลเกี่ยวกับฐานข้อมูล"""
        return {
            "total_identities": len(self.face_database),
            "total_embeddings": sum(len(embeddings) for embeddings in self.face_database.values()),
            "identities": list(self.face_database.keys())
        }