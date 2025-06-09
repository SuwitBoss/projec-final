# cSpell:disable
# mypy: ignore-errors
"""
Face Analysis Service
ระบบวิเคราะห์ใบหน้าแบบครบวงจร (Detection + Recognition)
Enhanced End-to-End Solution
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Any
import logging
import time
import asyncio

from .models import (
    FaceAnalysisResult, FaceResult, AnalysisConfig, 
    AnalysisMode, BatchAnalysisResult
)

# Import face detection service (assuming it exists)
try:
    from ..face_detection.face_detection_service import FaceDetectionService
except ImportError:
    FaceDetectionService = None

# Import face recognition service (assuming it exists)  
try:
    from ..face_recognition.face_recognition_service import FaceRecognitionService
    from ..face_recognition.models import FaceGallery
except ImportError:
    FaceRecognitionService = None
    FaceGallery = Dict[str, Any]

# Import VRAM manager
try:
    from ..common.vram_manager import VRAMManager
except ImportError:
    VRAMManager = None


class FaceAnalysisService:
    """
    Enhanced Face Analysis Service
    รวม Face Detection + Face Recognition ในระบบเดียว
    """
    
    def __init__(self, vram_manager: Any, config: Dict[str, Any]):
        self.vram_manager = vram_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
          # Initialize sub-services
        if FaceDetectionService:
            self.face_detection = FaceDetectionService(
                vram_manager, 
                config.get('detection', {})
            )
        else:
            self.face_detection = None
            
        if FaceRecognitionService:
            self.face_recognition = FaceRecognitionService(
                vram_manager,
                config.get('recognition', {})
            )
        else:
            self.face_recognition = None
            
        # Performance tracking
        self.stats: Dict[str, Any] = {
            'total_analyses': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'processing_times': [],
            'success_rates': []
        }
        
        self.logger.info("Face Analysis Service initialized")
    
    async def initialize(self) -> bool:
        """เริ่มต้นระบบ Face Analysis"""
        try:
            # Initialize face detection
            if self.face_detection:
                detection_init = await self.face_detection.initialize()
                if not detection_init:
                    self.logger.error("Failed to initialize face detection")
                    return False
            else:
                self.logger.warning("Face detection service not available")
            
            # Initialize face recognition  
            if self.face_recognition:
                recognition_init = await self.face_recognition.initialize()
                if not recognition_init:
                    self.logger.error("Failed to initialize face recognition")
                    return False
            else:
                self.logger.warning("Face recognition service not available")
            
            self.logger.info("Face Analysis Service ready")
            return True
            
        except Exception as e:
            self.logger.error(f"Face Analysis Service initialization failed: {e}")
            return False
    
    async def analyze_faces(self,
                           image: np.ndarray,
                           config: AnalysisConfig,
                           gallery: Optional[FaceGallery] = None) -> FaceAnalysisResult:
        """
        วิเคราะห์ใบหน้าครบวงจร
        
        Args:
            image: รูปภาพ (BGR format)
            config: การตั้งค่าการวิเคราะห์
            gallery: ฐานข้อมูลใบหน้าสำหรับจดจำ
            
        Returns:
            FaceAnalysisResult
        """
        start_time = time.time()
        detection_time = 0.0
        recognition_time = 0.0
        
        try:
            faces = []
            detection_model_used = None
            recognition_model_used = None
              # Step 1: Face Detection (ถ้าต้องการ)
            if config.mode in [AnalysisMode.DETECTION_ONLY, AnalysisMode.FULL_ANALYSIS]:
                if not self.face_detection:
                    raise RuntimeError("Face detection service not available")
                    
                detection_start = time.time()
                
                detection_result = await self.face_detection.detect_faces(
                    image,
                    model_name=config.detection_model,
                    min_face_size=config.min_face_size,
                    confidence_threshold=config.confidence_threshold,
                    max_faces=config.max_faces
                )
                
                detection_time = time.time() - detection_start
                detection_model_used = detection_result.model_used
                
                self.logger.info(f"Detection: {len(detection_result.faces)} faces in {detection_time:.3f}s")
                
                # แปลง detection results เป็น FaceResult
                for i, detected_face in enumerate(detection_result.faces):
                    face_result = FaceResult(
                        bbox=detected_face.bbox,
                        confidence=detected_face.confidence,
                        quality_score=detected_face.quality_score,
                        face_id=f"face_{i:03d}"
                    )
                    
                    # ตัดใบหน้าถ้าต้องการ
                    if config.return_face_crops or config.mode == AnalysisMode.FULL_ANALYSIS:
                        face_crop = self._extract_face_crop(image, detected_face.bbox)
                        if config.return_face_crops:
                            face_result.face_crop = face_crop
                    
                    faces.append(face_result)
              # Step 2: Face Recognition (ถ้าต้องการ)
            if config.mode in [AnalysisMode.FULL_ANALYSIS] and gallery and config.enable_gallery_matching:
                if not self.face_recognition:
                    self.logger.warning("Face recognition service not available")
                else:
                    recognition_start = time.time()
                
                # ประมวลผล recognition สำหรับใบหน้าที่มีคุณภาพดี
                quality_threshold = 60.0 if config.use_quality_based_selection else 0.0
                processable_faces = [f for f in faces if f.quality_score >= quality_threshold]
                
                if processable_faces:
                    recognition_results = await self._process_recognition(
                        image, processable_faces, config, gallery
                    )
                    
                    # อัปเดต face results ด้วยผลลัพธ์ recognition
                    for i, face_result in enumerate(faces):
                        if face_result in processable_faces:
                            idx = processable_faces.index(face_result)
                            if idx < len(recognition_results):
                                recognition_result = recognition_results[idx]
                                face_result.embedding = recognition_result.face_embedding
                                face_result.matches = recognition_result.matches
                                face_result.best_match = recognition_result.best_match
                
                recognition_time = time.time() - recognition_start
                
                if processable_faces:
                    recognition_model_used = processable_faces[0].embedding.model_used if processable_faces[0].embedding else None
                
                self.logger.info(f"Recognition: {len(processable_faces)} faces processed in {recognition_time:.3f}s")
            
            total_time = time.time() - start_time
              # สร้างผลลัพธ์
            result = FaceAnalysisResult(
                image_shape=(image.shape[0], image.shape[1], image.shape[2]),
                config=config,
                faces=faces,
                detection_time=detection_time,
                recognition_time=recognition_time,
                total_time=total_time,
                detection_model_used=detection_model_used,
                recognition_model_used=recognition_model_used
            )
            
            # อัปเดต statistics
            self._update_stats(result)
            
            self.logger.info(f"Analysis complete: {result.total_faces} faces, "
                           f"{result.identified_faces} identified in {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Face analysis failed: {e}")
            raise
    
    async def _process_recognition(self,
                                 image: np.ndarray,
                                 faces: List[FaceResult],
                                 config: AnalysisConfig,
                                 gallery: FaceGallery):
        """ประมวลผล Face Recognition สำหรับหลายใบหน้า"""
        recognition_tasks = []
        
        for face_result in faces:
            # ตัดใบหน้า
            face_crop = face_result.face_crop
            if face_crop is None:
                face_crop = self._extract_face_crop(image, face_result.bbox)
              # สร้าง task สำหรับ recognition
            if self.face_recognition:
                task = self.face_recognition.recognize_face(
                    face_crop,
                    gallery,
                    model_name=config.recognition_model,
                    top_k=config.gallery_top_k
                )
                recognition_tasks.append(task)
            else:
                # Create a dummy task that returns an error
                async def dummy_task():
                    raise RuntimeError("Face recognition service not available")
                recognition_tasks.append(dummy_task())
        
        # ประมวลผลแบบ parallel ถ้าเปิดใช้งาน
        if config.parallel_processing:
            results = await asyncio.gather(*recognition_tasks, return_exceptions=True)
        else:
            results = []
            for task in recognition_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Recognition task failed: {e}")
                    results.append(e)
        
        # กรองเฉพาะผลลัพธ์ที่สำเร็จ
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Recognition failed for face {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def _extract_face_crop(self, image: np.ndarray, bbox) -> np.ndarray:
        """ตัดใบหน้าจากรูปภาพ"""
        try:
            # ขยาย bbox เล็กน้อยเพื่อให้ได้ context
            h, w = image.shape[:2]
            
            # คำนวณการขยาย (15% ของขนาดหน้า)
            face_w = bbox.x2 - bbox.x1
            face_h = bbox.y2 - bbox.y1
            
            expand_w = int(face_w * 0.15)
            expand_h = int(face_h * 0.15)
            
            # ขยาย bbox
            x1 = max(0, bbox.x1 - expand_w)
            y1 = max(0, bbox.y1 - expand_h)
            x2 = min(w, bbox.x2 + expand_w)
            y2 = min(h, bbox.y2 + expand_h)
            
            # ตัดใบหน้า
            face_crop = image[y1:y2, x1:x2]
            
            # แปลงจาก BGR เป็น RGB สำหรับ face recognition
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            return face_rgb
            
        except Exception as e:
            self.logger.error(f"Failed to extract face crop: {e}")
            # Return a default face crop
            return np.zeros((112, 112, 3), dtype=np.uint8)
    
    async def compare_faces(self,
                           face_image1: np.ndarray,
                           face_image2: np.ndarray,
                           model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        เปรียบเทียบใบหน้า 2 ใบ
        
        Args:
            face_image1: รูปใบหน้าที่ 1 (RGB format)
            face_image2: รูปใบหน้าที่ 2 (RGB format)
            model_name: ชื่อโมเดลที่ต้องการใช้
              Returns:
            ผลลัพธ์การเปรียบเทียบ
        """
        start_time = time.time()
        
        try:
            if not self.face_recognition:
                raise RuntimeError("Face recognition service not available")
                
            # สกัด embeddings
            embedding1 = await self.face_recognition.extract_embedding(face_image1, model_name)
            embedding2 = await self.face_recognition.extract_embedding(face_image2, model_name)
            
            # เปรียบเทียบ
            comparison_result = self.face_recognition.compare_faces(
                embedding1.embedding,
                embedding2.embedding,
                embedding1.model_used
            )
            
            processing_time = time.time() - start_time
            
            return {
                'comparison': comparison_result.to_dict(),
                'embedding1': embedding1.to_dict(),
                'embedding2': embedding2.to_dict(),
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Face comparison failed: {e}")
            raise
    
    async def batch_analyze(self,
                           images: List[np.ndarray],
                           config: AnalysisConfig,
                           gallery: Optional[FaceGallery] = None) -> BatchAnalysisResult:
        """
        วิเคราะห์หลายรูปพร้อมกัน
        
        Args:
            images: รายการรูปภาพ
            config: การตั้งค่าการวิเคราะห์
            gallery: ฐานข้อมูลใบหน้า
        """
        start_time = time.time()
        
        try:
            # สร้าง tasks สำหรับแต่ละรูป
            analysis_tasks = []
            for i, image in enumerate(images):
                task = self.analyze_faces(image, config, gallery)
                analysis_tasks.append(task)
            
            # ประมวลผลแบบ parallel
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
              # กรองผลลัพธ์ที่สำเร็จ
            valid_results: List[FaceAnalysisResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Analysis failed for image {i}: {result}")
                else:
                    valid_results.append(result)
            
            # สรุปผลลัพธ์
            total_faces = sum(len(result.faces) for result in valid_results)
            
            # นับ unique identities
            all_identities = set()
            for result in valid_results:
                for face in result.faces:
                    if hasattr(face, 'has_identity') and face.has_identity:
                        all_identities.add(getattr(face, 'identity', ''))
            
            processing_time = time.time() - start_time
            
            return BatchAnalysisResult(
                results=valid_results,
                total_images=len(images),
                total_faces=total_faces,
                total_identities=len(all_identities),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            raise
    
    async def detect_and_recognize(self,
                                  image: np.ndarray,
                                  known_faces: Dict[str, np.ndarray],
                                  config: Optional[AnalysisConfig] = None) -> FaceAnalysisResult:
        """
        ตรวจจับและจดจำใบหน้าในขั้นตอนเดียว (Simplified API)
        
        Args:
            image: รูปภาพ
            known_faces: ฐานข้อมูลใบหน้า {identity_id: embedding}
            config: การตั้งค่า (ใช้ default ถ้าไม่ระบุ)
        """
        if config is None:
            config = AnalysisConfig(
                mode=AnalysisMode.FULL_ANALYSIS,
                enable_gallery_matching=True,
                use_quality_based_selection=True
            )
        
        # แปลง known_faces เป็น FaceGallery format
        gallery = {}
        for identity_id, embedding in known_faces.items():
            gallery[identity_id] = {
                'name': identity_id,                'embeddings': [embedding]
            }
        
        return await self.analyze_faces(image, config, gallery)
    
    def _update_stats(self, result: FaceAnalysisResult):
        """อัปเดต statistics"""
        self.stats['total_analyses'] += 1
        self.stats['total_faces_detected'] += result.total_faces
        self.stats['total_faces_recognized'] += result.identified_faces
        self.stats['processing_times'].append(result.total_time)
        
        if result.usable_faces > 0:
            success_rate = result.identified_faces / result.usable_faces
            self.stats['success_rates'].append(success_rate)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """ดึงสถิติประสิทธิภาพ"""
        stats = self.stats.copy()
        
        if self.stats['processing_times']:
            stats['average_processing_time'] = np.mean(self.stats['processing_times'])
            stats['total_processing_time'] = sum(self.stats['processing_times'])
            
        if self.stats['success_rates']:
            stats['average_success_rate'] = np.mean(self.stats['success_rates'])
        
        if self.face_detection:
            stats['detection_stats'] = self.face_detection.get_performance_stats()
        else:
            stats['detection_stats'] = {}
            
        if self.face_recognition:
            stats['recognition_stats'] = self.face_recognition.get_performance_stats()
        else:
            stats['recognition_stats'] = {}
        
        return stats
    
    async def get_available_models(self) -> Dict[str, Any]:
        """ดึงรายการโมเดลที่มีอยู่"""
        available_vram = await self.vram_manager.get_available_memory() if self.vram_manager else 0
        
        # Detection models
        detection_models = {}
        if self.face_detection and hasattr(self.face_detection, 'model_selector'):
            detection_models = self.face_detection.model_selector.get_performance_comparison()
          # Recognition models  
        recognition_models = {}
        recommendations = {}
        
        if self.face_recognition and hasattr(self.face_recognition, 'model_selector'):
            recognition_models = self.face_recognition.model_selector.get_performance_comparison()
            recommendations = self.face_recognition.model_selector.recommend_models(available_vram)
        
        return {
            'available_vram_mb': available_vram,
            'detection_models': detection_models,
            'recognition_models': recognition_models,            'recommendations': recommendations
        }
    
    async def switch_models(self,
                           detection_model: Optional[str] = None,
                           recognition_model: Optional[str] = None) -> Dict[str, bool]:
        """เปลี่ยนโมเดลที่ใช้งาน"""
        results = {}
        
        if detection_model and self.face_detection:
            success = await self.face_detection.switch_model(detection_model)
            results['detection'] = success
        elif detection_model:
            results['detection'] = False
        
        if recognition_model and self.face_recognition:
            success = await self.face_recognition.switch_model(recognition_model)
            results['recognition'] = success
        elif recognition_model:
            results['recognition'] = False
        
        return results
    
    async def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        try:
            if self.face_detection:
                await self.face_detection.cleanup()
            if self.face_recognition:
                await self.face_recognition.cleanup()
            
            self.logger.info("Face Analysis Service cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def create_gallery_from_embeddings(self, 
                                     embeddings_dict: Dict[str, Any]) -> FaceGallery:
        """
        สร้าง FaceGallery จาก embeddings dictionary
        
        Args:
            embeddings_dict: {identity_id: {'name': str, 'embeddings': List[np.ndarray]}}
        """
        gallery = {}
        for identity_id, data in embeddings_dict.items():
            gallery[identity_id] = {
                'name': data.get('name', identity_id),
                'embeddings': data.get('embeddings', [])
            }
        return gallery
