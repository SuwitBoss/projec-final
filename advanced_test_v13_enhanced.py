#!/usr/bin/env python3
"""
ระบบ Face Recognition ขั้นสูง v13 Enhanced - รองรับ Ensemble System
- Ultra Quality Enhancement Pipeline with Super Resolution
- Advanced preprocessing และ Multi-scale processing
- Quality-aware face cropping
- Multi-tier Threshold System
- Cross-person validation
- Context-aware recognition (single vs group photos)
- **NEW: Face Recognition Ensemble System Support**
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import sys

# เพิ่ม path เพื่อ import modules
sys.path.append('src')

from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService, RecognitionConfig
from src.ai_services.face_recognition.ensemble_face_recognition_service import EnsembleFaceRecognitionService, EnsembleConfig
from src.ai_services.face_detection.face_detection_service import FaceDetectionService
from src.ai_services.face_recognition.models import ModelType
from src.ai_services.common.vram_manager import VRAMManager

class UltraQualityEnhancer:
    """ระบบปรับปรุงคุณภาพภาพขั้นสูง"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Super Resolution parameters
        self.sr_scale_factor = 2  # ขยายขนาด 2 เท่า
        self.target_face_size = 224  # ขนาดใบหน้าเป้าหมาย (เพิ่มจาก 160)
        
    def enhance_image_ultra_quality(self, image: np.ndarray) -> np.ndarray:
        """Ultra Quality Enhancement Pipeline"""
        try:
            original_height, original_width = image.shape[:2]
            self.logger.debug(f"Original size: {original_width}x{original_height}")
            
            # === STAGE 1: Noise Reduction ===
            # 1.1 Non-local means denoising (ลดสัญญาณรบกวน)
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # === STAGE 2: Super Resolution (ถ้าภาพเล็กกว่า threshold) ===
            if min(original_width, original_height) < 800:  # ถ้าภาพเล็กกว่า 800px
                self.logger.debug("Applying super resolution...")
                denoised = self.apply_super_resolution(denoised)
            
            # === STAGE 3: Advanced Enhancement ===
            # 3.1 CLAHE with optimized parameters
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # ปรับพารามิเตอร์
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3.2 Bilateral filter (edge-preserving smoothing)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 3.3 Unsharp masking (advanced sharpening)
            enhanced = self.unsharp_mask(enhanced, sigma=1.0, strength=1.5)
            
            # 3.4 Adaptive gamma correction
            enhanced = self.adaptive_gamma_correction(enhanced)
            
            # === STAGE 4: Color Enhancement ===
            # 4.1 Saturation boost (เพิ่มความอิ่มตัวของสี)
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)  # เพิ่ม saturation 20%
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 4.2 Contrast enhancement
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
            
            final_height, final_width = enhanced.shape[:2]
            self.logger.debug(f"Enhanced size: {final_width}x{final_height}")
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Enhancement failed: {e}")
            return image  # ถ้าล้มเหลวให้ใช้ภาพต้นฉบับ
    
    def apply_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """Apply Super Resolution using EDSR-like interpolation"""
        try:
            height, width = image.shape[:2]
            new_width = width * self.sr_scale_factor
            new_height = height * self.sr_scale_factor
            
            # ใช้ INTER_CUBIC สำหรับการขยายที่มีคุณภาพสูง
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # เพิ่ม additional sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            
            # ผสมภาพต้นฉบับกับภาพที่ sharp
            result = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception:
            return image
    
    def unsharp_mask(self, image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        """Apply unsharp masking for better edge definition"""
        try:
            # สร้าง blurred version
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # สร้าง sharpened image
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            
            return sharpened
            
        except Exception:
            return image
    
    def adaptive_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive gamma correction based on image histogram"""
        try:
            # แปลงเป็น grayscale เพื่อวิเคราะห์
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # คำนวณ gamma value จาก histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            
            # หา gamma ที่เหมาะสม
            cumsum = np.cumsum(hist_normalized)
            median_idx = np.where(cumsum >= 0.5)[0][0]
            gamma = np.log(0.5) / np.log(median_idx / 255.0) if median_idx > 0 else 1.0
            gamma = np.clip(gamma, 0.5, 2.0)  # จำกัดช่วง gamma
            
            # สร้าง lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction
            corrected = cv2.LUT(image, table)
            
            return corrected
            
        except Exception:
            return image
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """ประเมินคุณภาพภาพโดยรวม"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000, 1.0)
            
            # 2. Brightness distribution
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # 3. Contrast (standard deviation)
            contrast = np.std(gray)
            contrast_score = min(contrast / 64, 1.0)
            
            # 4. Noise level (high frequency content)
            noise_level = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            noise_score = 1.0 - min(noise_level / 100, 1.0)
            
            # 5. Overall quality
            overall_quality = (sharpness_score * 0.4 + brightness_score * 0.2 + 
                             contrast_score * 0.2 + noise_score * 0.2)
            
            return {
                'sharpness': sharpness_score,
                'brightness': brightness_score,
                'contrast': contrast_score,
                'noise': noise_score,
                'overall': overall_quality
            }
            
        except Exception:
            return {'overall': 0.5}
    
    def crop_face_ultra_quality(self, image: np.ndarray, bbox, target_size: int = 224) -> np.ndarray:
        """ตัดใบหน้าด้วยคุณภาพสูงสุด"""
        try:
            height, width = image.shape[:2]
            
            # ขยายขอบเขตเล็กน้อยสำหรับใบหน้า
            margin = 0.3  # เพิ่ม margin สำหรับคุณภาพที่ดีขึ้น
            x1 = max(0, int(bbox.x1 - bbox.width * margin))
            y1 = max(0, int(bbox.y1 - bbox.height * margin))
            x2 = min(width, int(bbox.x2 + bbox.width * margin))
            y2 = min(height, int(bbox.y2 + bbox.height * margin))
            
            # ตัดใบหน้า
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                # Fallback ถ้าการขยายทำให้เกิดปัญหา
                x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
                x2, y2 = min(width, int(bbox.x2)), min(height, int(bbox.y2))
                face_crop = image[y1:y2, x1:x2]
            
            # Resize to target size with high quality interpolation
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (target_size, target_size), 
                                     interpolation=cv2.INTER_LANCZOS4)
                
                # Apply additional enhancement to face crop
                face_crop = self.enhance_face_crop(face_crop)
            
            return face_crop
            
        except Exception as e:
            self.logger.error(f"❌ Error cropping face: {e}")
            return np.array([])
    
    def enhance_face_crop(self, face_crop: np.ndarray) -> np.ndarray:
        """ปรับปรุงคุณภาพของใบหน้าที่ตัดแล้ว"""
        try:
            # 1. Histogram equalization
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 2. Slight sharpening
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 3. Noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            return enhanced
            
        except Exception:
            return face_crop


class AdvancedRealImageTestSystemV13Enhanced:
    """ระบบทดสอบ Face Recognition ขั้นสูงพร้อม Ensemble Support"""
    
    def __init__(self, use_ensemble: bool = True):
        self.setup_logging()
        self.output_dir = Path("output/advanced_real_image_test_v13_enhanced")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === NEW: Ultra Quality Enhancer ===
        self.ultra_enhancer = UltraQualityEnhancer()
        self.logger.info("🚀 Ultra Quality Enhancement initialized")
        
        # === NEW: Ensemble Support ===
        self.use_ensemble = use_ensemble
        
        if self.use_ensemble:
            self.logger.info("🎯 Ensemble mode enabled - using AdaFace + FaceNet + ArcFace")
            self.ensemble_config = EnsembleConfig(
                adaface_weight=0.25,   # 25% ตามเอกสาร
                facenet_weight=0.50,   # 50% ตามเอกสาร
                arcface_weight=0.25,   # 25% ตามเอกสาร
                ensemble_threshold=0.20,
                enable_gpu_optimization=True,
                quality_threshold=0.2
            )
        else:
            self.logger.info("🔧 Individual model mode - using FaceNet")
            # === ADVANCED: Multi-tier Threshold System ===
            self.config = RecognitionConfig(
                similarity_threshold=0.60,  # เพิ่มจาก 0.55 เป็น 0.60 (กลางๆ)
                unknown_threshold=0.55,     # เพิ่มจาก 0.50 เป็น 0.55
                quality_threshold=0.2,
                preferred_model=ModelType.FACENET
            )
        
        # === NEW: Multi-tier Thresholds ===
        self.thresholds = {
            'high_confidence': 0.85,    # มั่นใจสูง
            'medium_confidence': 0.70,  # มั่นใจปานกลาง  
            'low_confidence': 0.55,     # มั่นใจต่ำ (เริ่มต้น)
            'unknown_boundary': 0.50    # เส้นแบ่ง unknown
        }
        
        # === VRAM Management ===
        vram_config = {
            "total_vram_mb": 12288,  # RTX 3060 12GB
            "reserved_system_mb": 2048,
            "model_vram_estimates": {
                "yolov9c-face": 512 * 1024 * 1024,
                "yolov9e-face": 2048 * 1024 * 1024,
                "yolov11m-face": 2 * 1024 * 1024 * 1024,
                "adaface": 89 * 1024 * 1024,
                "arcface": 249 * 1024 * 1024,
                "facenet": 249 * 1024 * 1024,
            }
        }
        self.vram_manager = VRAMManager(vram_config)
        
        # === Services ===
        if self.use_ensemble:
            self.face_service = EnsembleFaceRecognitionService(self.ensemble_config, self.vram_manager)
        else:
            self.face_service = FaceRecognitionService(self.config, self.vram_manager)
        
        # การตั้งค่า detection เหมือนระบบเดิม
        detection_config = {
            "yolov9c_model_path": "model/face-detection/yolov9c-face-lindevs.onnx",
            "yolov9e_model_path": "model/face-detection/yolov9e-face-lindevs.onnx",
            "yolov11m_model_path": "model/face-detection/yolov11m-face.pt",
            "max_usable_faces_yolov9": 8,
            "min_agreement_ratio": 0.7,
            "min_quality_threshold": 60,
            "conf_threshold": 0.15,
            "iou_threshold": 0.4,
            "img_size": 640
        }
        self.detection_service = FaceDetectionService(self.vram_manager, detection_config)
        
        # เก็บ embeddings ของคนที่ลงทะเบียน
        self.registered_people = {}
        self.dynamic_embeddings = {}
        
        # === NEW: Disable dynamic embeddings ===
        self.enable_dynamic_embeddings = False  # ปิดใช้งานต่อ
        
        # === NEW: Reference Image Tracking ===
        self.reference_images = set()
        
        # === NEW: Context tracking ===
        self.recognition_stats = {
            'total_faces': 0,
            'false_positives_prevented': 0,
            'cross_person_rejections': 0,
            'group_photo_rejections': 0,
            'high_confidence_matches': 0
        }
        
        self._initialized = False
        
    def setup_logging(self):
        """ตั้งค่า logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('advanced_real_image_test_v13_enhanced.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def initialize_services(self):
        """Initialize all services"""
        if not self._initialized:
            await self.face_service.initialize()
            await self.detection_service.initialize()
            self._initialized = True
            self.logger.info("✅ Services initialized successfully")

    async def enroll_person(self, image_path: str, person_name: str) -> bool:
        """ลงทะเบียนบุคคลหนึ่งคน"""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"❌ ไม่พบไฟล์: {image_path}")
                return False
                
            # เก็บรายชื่อ reference images
            filename = os.path.basename(image_path)
            self.reference_images.add(filename)
                
            # อ่านและปรับปรุงภาพ
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"❌ ไม่สามารถอ่านภาพได้: {image_path}")
                return False
                
            # === ULTRA QUALITY ENHANCEMENT ===
            enhanced_image = self.ultra_enhancer.enhance_image_ultra_quality(image)
            
            # ประเมินคุณภาพ
            quality_metrics = self.ultra_enhancer.assess_image_quality(enhanced_image)
            self.logger.debug(f"Image quality: {quality_metrics}")
            
            # ตรวจหาใบหน้า
            detection_result = await self.detection_service.detect_faces(enhanced_image)
            
            if not detection_result.faces:
                self.logger.warning(f"⚠️ ไม่พบใบหน้าใน: {image_path}")
                return False
                
            # ใช้ใบหน้าที่มี confidence สูงสุด
            best_face = max(detection_result.faces, key=lambda f: f.bbox.confidence)
            
            # === ULTRA QUALITY FACE CROPPING ===
            face_crop = self.ultra_enhancer.crop_face_ultra_quality(
                enhanced_image, best_face.bbox, target_size=224
            )
            
            if face_crop.size == 0:
                self.logger.error(f"❌ ไม่สามารถ crop ใบหน้าได้: {image_path}")
                return False
            
            if self.use_ensemble:
                # ใช้ Ensemble service
                success = await self.face_service.add_face_to_database(
                    person_name, face_crop,
                    metadata={
                        'source_image': image_path,
                        'detection_confidence': best_face.bbox.confidence,
                        'quality_metrics': quality_metrics,
                        'enhancement_applied': True
                    }
                )
            else:
                # ใช้ Individual service
                embedding = await self.face_service.extract_embedding(face_crop)
                if embedding is None:
                    self.logger.error(f"❌ ไม่สามารถสร้าง embedding ได้: {image_path}")
                    return False
                    
                # เก็บ embedding
                if person_name not in self.registered_people:
                    self.registered_people[person_name] = []
                    
                self.registered_people[person_name].append({
                    'embedding': embedding.vector,
                    'source_image': image_path,
                    'quality': best_face.bbox.confidence,
                    'bbox': best_face.bbox,
                    'enrollment_time': datetime.now().isoformat(),
                    'is_reference': True
                })
                success = True
            
            if success:
                self.logger.info(f"✅ ลงทะเบียน {person_name} จาก {image_path} สำเร็จ (Quality: {best_face.bbox.confidence:.3f})")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ เกิดข้อผิดพลาดในการลงทะเบียน {image_path}: {e}")
            return False

    async def enroll_reference_images(self) -> bool:
        """ลงทะเบียนภาพอ้างอิงทั้งหมด"""
        mode_text = "Ensemble System" if self.use_ensemble else "Individual Model"
        self.logger.info(f"📝 เริ่มการลงทะเบียนภาพอ้างอิงด้วย {mode_text}...")
        
        # รายการไฟล์อ้างอิง - boss_01-10 และ night_01-10
        reference_files = []
        
        # Boss images (boss_01 ถึง boss_10)
        for i in range(1, 11):
            reference_files.append((f"test_images/boss_{i:02d}.jpg", "Boss"))
        
        # Night images (night_01 ถึง night_10)  
        for i in range(1, 11):
            reference_files.append((f"test_images/night_{i:02d}.jpg", "Night"))
        
        total_registered = 0
        
        for image_path, person_name in reference_files:
            if await self.enroll_person(image_path, person_name):
                total_registered += 1
                
        # แสดงสรุป
        self.logger.info("=" * 50)
        self.logger.info(f"📊 สรุปการลงทะเบียน {mode_text}:")
        
        if self.use_ensemble:
            # แสดงสถิติ ensemble
            ensemble_stats = self.face_service.get_statistics()
            for person_name in ['Boss', 'Night']:
                person_data = self.face_service.face_database.get(person_name, [])
                self.logger.info(f"   👤 {person_name}: {len(person_data)} ภาพ")
            
            self.logger.info(f"   📈 รวมทั้งหมด: {total_registered} ภาพ")
            self.logger.info(f"   🎯 Ensemble weights: AdaFace 25%, FaceNet 50%, ArcFace 25%")
            self.logger.info(f"   🔧 Model success rates:")
            for model, stats in ensemble_stats['model_success_rates'].items():
                rate = stats.get('success_rate', 0) * 100 if stats['total'] > 0 else 0
                self.logger.info(f"      - {model.upper()}: {rate:.1f}%")
        else:
            # แสดงสถิติ individual
            for person_name, embeddings in self.registered_people.items():
                self.logger.info(f"   👤 {person_name}: {len(embeddings)} ภาพ")
            self.logger.info(f"   📈 รวมทั้งหมด: {total_registered} ภาพ")
            self.logger.info(f"   🔧 Model: FaceNet VGGFace2")
        
        self.logger.info(f"   🎯 Face crop size: 224x224 (Ultra Quality)")
        
        return total_registered > 0

    async def recognize_face_in_image(self, image_path: str) -> List[Dict[str, Any]]:
        """จดจำใบหน้าในภาพหนึ่งใบ"""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"❌ ไม่พบไฟล์: {image_path}")
                return []
                
            # อ่านและปรับปรุงภาพ
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"❌ ไม่สามารถอ่านภาพได้: {image_path}")
                return []
                
            # === ULTRA QUALITY ENHANCEMENT ===
            enhanced_image = self.ultra_enhancer.enhance_image_ultra_quality(image)
            
            # ตรวจหาใบหน้า
            detection_result = await self.detection_service.detect_faces(enhanced_image)
            
            if not detection_result.faces:
                self.logger.info(f"ℹ️ ไม่พบใบหน้าใน: {os.path.basename(image_path)}")
                return []
            
            # ประมวลผลใบหน้าทั้งหมด
            face_results = []
            
            for i, detected_face in enumerate(detection_result.faces):
                try:
                    # === ULTRA QUALITY FACE CROPPING ===
                    face_crop = self.ultra_enhancer.crop_face_ultra_quality(
                        enhanced_image, detected_face.bbox, target_size=224
                    )
                    
                    if face_crop.size == 0:
                        continue
                    
                    # จดจำใบหน้า
                    if self.use_ensemble:
                        recognition_result = await self.face_service.recognize_face(face_crop)
                    else:
                        embedding = await self.face_service.extract_embedding(face_crop)
                        if embedding is None:
                            continue
                        recognition_result = await self.individual_recognize_face(embedding)
                    
                    # เก็บผลลัพธ์
                    face_result = {
                        'face_id': i,
                        'bbox': {
                            'x1': detected_face.bbox.x1,
                            'y1': detected_face.bbox.y1,
                            'x2': detected_face.bbox.x2,
                            'y2': detected_face.bbox.y2,
                            'confidence': detected_face.bbox.confidence
                        },
                        'recognition': {
                            'identity': recognition_result.best_match.identity_id if recognition_result.best_match else "Unknown",
                            'confidence': recognition_result.confidence,
                            'similarity': recognition_result.best_match.similarity if recognition_result.best_match else 0.0,
                            'is_known': recognition_result.is_known,
                            'processing_time': recognition_result.processing_time
                        },
                        'quality_metrics': self.ultra_enhancer.assess_image_quality(face_crop)
                    }
                    
                    face_results.append(face_result)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing face {i}: {e}")
                    continue
            
            self.logger.info(f"🔍 ประมวลผล {len(face_results)} ใบหน้าใน {os.path.basename(image_path)}")
            
            return face_results
            
        except Exception as e:
            self.logger.error(f"❌ เกิดข้อผิดพลาดในการจดจำใบหน้า {image_path}: {e}")
            return []

    async def individual_recognize_face(self, embedding) -> Any:
        """จดจำใบหน้าสำหรับ individual model (ถ้าไม่ใช้ ensemble)"""
        try:
            # สร้าง mock result object ที่เข้ากันได้
            class MockMatch:
                def __init__(self, identity_id, similarity):
                    self.identity_id = identity_id
                    self.similarity = similarity
            
            class MockResult:
                def __init__(self):
                    self.best_match = None
                    self.confidence = 0.0
                    self.is_known = False
                    self.processing_time = 0.0
            
            result = MockResult()
            
            # หาความคล้าย
            best_similarity = 0.0
            best_person = None
            
            for person_name, person_embeddings in self.registered_people.items():
                for emb_data in person_embeddings:
                    similarity = np.dot(embedding.vector, emb_data['embedding'])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_person = person_name
            
            # ตรวจสอบ threshold
            threshold = self.config.similarity_threshold
            if best_similarity >= threshold:
                result.best_match = MockMatch(best_person, best_similarity)
                result.confidence = best_similarity / threshold
                result.is_known = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Individual recognition failed: {e}")
            class MockResult:
                def __init__(self):
                    self.best_match = None
                    self.confidence = 0.0
                    self.is_known = False
                    self.processing_time = 0.0
            return MockResult()

    async def run_comprehensive_test(self):
        """รันการทดสอบแบบครบถ้วน"""
        try:
            mode_text = "Ensemble System" if self.use_ensemble else "Individual Model (FaceNet)"
            self.logger.info(f"🚀 เริ่มการทดสอบ {mode_text}")
            self.logger.info("=" * 80)
            
            # Initialize services
            await self.initialize_services()
            
            # Enroll reference images
            enrollment_success = await self.enroll_reference_images()
            
            if not enrollment_success:
                self.logger.error("❌ การลงทะเบียนล้มเหลว")
                return
            
            # Test images
            test_files = [
                "test_images/boss_01.jpg",
                "test_images/boss_05.jpg", 
                "test_images/boss_10.jpg",
                "test_images/night_01.jpg",
                "test_images/night_05.jpg",
                "test_images/night_10.jpg",
                "test_images/boss_11.jpg",  # Unknown
                "test_images/boss_glass02.jpg",  # Modified
            ]
            
            all_results = []
            
            for test_file in test_files:
                if os.path.exists(test_file):
                    self.logger.info(f"🧪 ทดสอบ: {os.path.basename(test_file)}")
                    results = await self.recognize_face_in_image(test_file)
                    all_results.extend(results)
                    
                    for result in results:
                        identity = result['recognition']['identity']
                        confidence = result['recognition']['confidence']
                        self.logger.info(f"   👤 ผลลัพธ์: {identity} (confidence: {confidence:.3f})")
                else:
                    self.logger.warning(f"⚠️ ไม่พบไฟล์: {test_file}")
            
            # Save results
            self.save_test_results(all_results)
            
            # Show summary
            self.show_test_summary(all_results)
            
            self.logger.info("=" * 80)
            self.logger.info(f"✅ การทดสอบ {mode_text} เสร็จสมบูรณ์!")
            
        except Exception as e:
            self.logger.error(f"❌ การทดสอบล้มเหลว: {e}")
            raise

    def save_test_results(self, results: List[Dict[str, Any]]):
        """บันทึกผลลัพธ์การทดสอบ"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_suffix = "ensemble" if self.use_ensemble else "individual"
            results_file = self.output_dir / f"test_results_{mode_suffix}_{timestamp}.json"
            
            output_data = {
                'test_info': {
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'ensemble' if self.use_ensemble else 'individual',
                    'model_config': self.ensemble_config.__dict__ if self.use_ensemble else self.config.__dict__,
                    'total_faces_tested': len(results)
                },
                'results': results
            }
            
            if self.use_ensemble:
                output_data['ensemble_statistics'] = self.face_service.get_statistics()
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"💾 ผลลัพธ์บันทึกแล้วที่: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")

    def show_test_summary(self, results: List[Dict[str, Any]]):
        """แสดงสรุปผลการทดสอบ"""
        try:
            if not results:
                self.logger.info("ℹ️ ไม่มีผลลัพธ์ที่จะแสดง")
                return
            
            # นับผลลัพธ์
            identity_counts = {}
            confidence_scores = []
            known_faces = 0
            
            for result in results:
                identity = result['recognition']['identity']
                confidence = result['recognition']['confidence']
                is_known = result['recognition']['is_known']
                
                identity_counts[identity] = identity_counts.get(identity, 0) + 1
                confidence_scores.append(confidence)
                
                if is_known:
                    known_faces += 1
            
            self.logger.info("📊 สรุปผลการทดสอบ:")
            self.logger.info(f"   🔢 จำนวนใบหน้าทั้งหมด: {len(results)}")
            self.logger.info(f"   ✅ ใบหน้าที่รู้จัก: {known_faces}")
            self.logger.info(f"   ❓ ใบหน้าที่ไม่รู้จัก: {len(results) - known_faces}")
            
            if confidence_scores:
                avg_confidence = np.mean(confidence_scores)
                self.logger.info(f"   📈 ความมั่นใจเฉลี่ย: {avg_confidence:.3f}")
            
            self.logger.info("   👥 การจดจำตามบุคคล:")
            for identity, count in identity_counts.items():
                percentage = (count / len(results)) * 100
                self.logger.info(f"      - {identity}: {count} ครั้ง ({percentage:.1f}%)")
            
            if self.use_ensemble:
                # แสดงสถิติ ensemble
                ensemble_stats = self.face_service.get_statistics()
                self.logger.info("   🔧 สถิติ Ensemble:")
                self.logger.info(f"      - Total extractions: {ensemble_stats['total_extractions']}")
                self.logger.info(f"      - Total recognitions: {ensemble_stats['total_recognitions']}")
                if ensemble_stats['ensemble_processing_times']:
                    avg_time = np.mean(ensemble_stats['ensemble_processing_times'])
                    self.logger.info(f"      - Average processing time: {avg_time:.3f}s")
                
        except Exception as e:
            self.logger.error(f"❌ Error showing summary: {e}")


async def main():
    """ฟังก์ชันหลัก"""
    try:
        print("🎯 Face Recognition Test System V13 Enhanced")
        print("=" * 50)
        print("เลือกโหมดการทดสอบ:")
        print("1. Ensemble System (AdaFace + FaceNet + ArcFace)")
        print("2. Individual Model (FaceNet)")
        print("3. ทดสอบทั้งสองแบบ")
        
        choice = input("กรุณาเลือก (1/2/3): ").strip()
        
        if choice == "1":
            # ทดสอบ Ensemble
            test_system = AdvancedRealImageTestSystemV13Enhanced(use_ensemble=True)
            await test_system.run_comprehensive_test()
            
        elif choice == "2":
            # ทดสอบ Individual
            test_system = AdvancedRealImageTestSystemV13Enhanced(use_ensemble=False)
            await test_system.run_comprehensive_test()
            
        elif choice == "3":
            # ทดสอบทั้งสองแบบ
            print("\n🔧 ทดสอบ Individual Model ก่อน...")
            test_system1 = AdvancedRealImageTestSystemV13Enhanced(use_ensemble=False)
            await test_system1.run_comprehensive_test()
            
            print("\n🎯 ทดสอบ Ensemble System...")
            test_system2 = AdvancedRealImageTestSystemV13Enhanced(use_ensemble=True)
            await test_system2.run_comprehensive_test()
            
        else:
            print("❌ ตัวเลือกไม่ถูกต้อง")
        
    except Exception as e:
        logging.error(f"❌ การทดสอบล้มเหลว: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
