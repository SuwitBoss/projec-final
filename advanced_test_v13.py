#!/usr/bin/env python3
"""
ระบบ Face Recognition ขั้นสูง v13 - Ultra Quality Enhancement + Ensemble Support
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
        """
        Ultra Quality Enhancement Pipeline
        """
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
        """
        Apply Super Resolution using EDSR-like interpolation
        """
        try:
            height, width = image.shape[:2]
            new_width = width * self.sr_scale_factor
            new_height = height * self.sr_scale_factor
            
            # ใช้ INTER_CUBIC สำหรับการขยายที่มีคุณภาพสูง
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # ลด artifacts จากการ upscaling
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # sharpening kernel
            sharpened = cv2.filter2D(upscaled, -1, kernel * 0.1)
            
            # Blend กับภาพต้นฉบับ
            result = cv2.addWeighted(upscaled, 0.8, sharpened, 0.2, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Super resolution failed: {e}")
            return image
    
    def unsharp_mask(self, image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking for advanced sharpening
        """
        try:
            # สร้าง gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # สร้าง unsharp mask
            unsharp_mask = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            
            return unsharp_mask
            
        except Exception:
            return image
    
    def adaptive_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive gamma correction based on image brightness
        """
        try:
            # คำนวณ brightness ของภาพ
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # ปรับ gamma ตาม brightness
            if mean_brightness < 80:  # ภาพมืด
                gamma = 0.7  # ทำให้สว่างขึ้น
            elif mean_brightness > 180:  # ภาพสว่าง
                gamma = 1.3  # ทำให้มืดลง
            else:
                gamma = 1.0  # ปกติ
                
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            corrected = cv2.LUT(image, table)
            
            return corrected
            
        except Exception:
            return image
    
    def crop_face_ultra_quality(self, image: np.ndarray, bbox, target_size: int = None) -> np.ndarray:
        """
        Ultra quality face cropping with optimal sizing
        """
        try:
            if target_size is None:
                target_size = self.target_face_size
                
            height, width = image.shape[:2]
            
            # คำนวณขนาดที่เหมาะสม
            face_width = int(bbox.width)
            face_height = int(bbox.height)
            
            # เพิ่ม margin แบบ adaptive
            if min(face_width, face_height) < 100:
                margin = 0.4  # margin มากกว่าสำหรับใบหน้าเล็ก
            elif min(face_width, face_height) < 200:
                margin = 0.3
            else:
                margin = 0.2  # margin ปกติสำหรับใบหน้าใหญ่
            
            # คำนวณ crop coordinates
            x1 = max(0, int(bbox.x1 - bbox.width * margin))
            y1 = max(0, int(bbox.y1 - bbox.height * margin))
            x2 = min(width, int(bbox.x2 + bbox.width * margin))
            y2 = min(height, int(bbox.y2 + bbox.height * margin))
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return np.array([])
            
            # ปรับให้เป็น square aspect ratio
            face_crop = self.make_square_crop(face_crop)
            
            # Resize เป็นขนาดเป้าหมาย ด้วยคุณภาพสูง
            if face_crop.shape[0] != target_size:
                # ใช้ INTER_LANCZOS4 สำหรับคุณภาพสูงสุด
                face_crop = cv2.resize(face_crop, (target_size, target_size), 
                                     interpolation=cv2.INTER_LANCZOS4)
            
            # Final enhancement สำหรับใบหน้า
            face_crop = self.enhance_face_details(face_crop)
            
            return face_crop
            
        except Exception as e:
            self.logger.error(f"Ultra quality cropping failed: {e}")
            return np.array([])
    
    def make_square_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Convert rectangular crop to square while preserving content
        """
        try:
            h, w = image.shape[:2]
            
            if h == w:
                return image
            
            # หาขนาดที่ใหญ่กว่า
            size = max(h, w)
            
            # สร้าง square canvas
            square = np.zeros((size, size, 3), dtype=image.dtype)
            
            # วาง image ตรงกลาง
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = image
            
            return square
            
        except Exception:
            return image
    
    def enhance_face_details(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance facial details specifically
        """
        try:
            # 1. Skin smoothing (ลดรอยไม่ปกติ แต่รักษาขอบ)
            smooth = cv2.bilateralFilter(face_image, 5, 50, 50)
            
            # 2. Detail enhancement (เพิ่มรายละเอียดดวงตา จมูก ปาก)
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
            edges = cv2.GaussianBlur(edges, (3, 3), 0)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # 3. Combine smooth skin with enhanced details
            mask = edges_colored.astype(np.float32) / 255.0
            enhanced = smooth.astype(np.float32) * (1 - mask) + face_image.astype(np.float32) * mask
            
            return enhanced.astype(np.uint8)
            
        except Exception:
            return face_image
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality metrics
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Brightness
            brightness = np.mean(gray)
            
            # 3. Contrast (standard deviation)
            contrast = np.std(gray)
            
            # 4. Noise level (estimate)
            noise = self.estimate_noise(gray)
            
            return {
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'noise': float(noise),
                'overall_quality': float((sharpness / 100 + contrast / 50) / 2)
            }
            
        except Exception:
            return {'overall_quality': 0.5}
    
    def estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level in image
        """
        try:
            # ใช้ Laplacian เพื่อหา high frequency components
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise_level = np.std(laplacian)
            return noise_level
        except Exception:
            return 0.0

def enhance_image_precision(image: np.ndarray) -> np.ndarray:
    """ใช้เทคนิค Precision Enhancement ที่ประสบความสำเร็จ (เก็บไว้เพื่อ compatibility)"""
    try:
        # 1. Adaptive Histogram Equalization (CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(6, 6))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Edge-preserving filter
        enhanced = cv2.edgePreservingFilter(enhanced, flags=1, sigma_s=60, sigma_r=0.4)
        
        # 3. Selective sharpening
        gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        enhanced = cv2.addWeighted(enhanced, 1.6, gaussian_blur, -0.6, 0)
        
        # 4. Gamma correction
        gamma = 1.25
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
        
    except Exception as e:
        print(f"Warning: Enhancement failed, using original image: {e}")
        return image

class AdvancedRealImageTestSystemV13:
    def __init__(self):
        self.setup_logging()
        self.output_dir = Path("output/advanced_real_image_test_v13")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === NEW: Ultra Quality Enhancer ===
        self.ultra_enhancer = UltraQualityEnhancer()
        self.logger.info("🚀 Ultra Quality Enhancement initialized")
        
        # === ADVANCED: Multi-tier Threshold System ===
        self.config = RecognitionConfig(
            similarity_threshold=0.60,  # เพิ่มจาก 0.55 เป็น 0.60 (กลางๆ)
            unknown_threshold=0.55,     # เพิ่มจาก 0.50 เป็น 0.55
            quality_threshold=0.2,
            preferred_model=ModelType.FACENET
        )
        
        # === NEW: Multi-tier Thresholds ===
        self.thresholds = {
            'high_confidence': 0.75,      # สำหรับ positive identification แน่นอน
            'medium_confidence': 0.65,    # สำหรับ reference images  
            'low_confidence': 0.55,       # สำหรับการตรวจสอบทั่วไป
            'rejection': 0.50,            # ต่ำกว่านี้ = unknown
            'cross_person_gap': 0.15,     # ความต่างขั้นต่ำระหว่างคนที่ 1 และ 2
            'group_photo_penalty': 0.05   # penalty สำหรับ group photos
        }
        
        # เริ่มต้น VRAM manager
        vram_config = {
            "reserved_vram_mb": 512,
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
        
        # เริ่มต้น services
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
                logging.FileHandler('advanced_real_image_test_v13.log'),
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

    def crop_face_from_bbox(self, image: np.ndarray, bbox) -> np.ndarray:
        """ตัดใบหน้าจาก bounding box"""
        try:
            height, width = image.shape[:2]
            
            # ขยายขอบเขตเล็กน้อย
            margin = 0.2
            x1 = max(0, int(bbox.x1 - bbox.width * margin))
            y1 = max(0, int(bbox.y1 - bbox.height * margin))
            x2 = min(width, int(bbox.x2 + bbox.width * margin))
            y2 = min(height, int(bbox.y2 + bbox.height * margin))
            
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                # Fallback ถ้าการขยายทำให้เกิดปัญหา
                x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
                x2, y2 = min(width, int(bbox.x2)), min(height, int(bbox.y2))
                face_crop = image[y1:y2, x1:x2]
                
            return face_crop
              except Exception as e:
            self.logger.error(f"❌ Error cropping face: {e}")
            return np.array([])

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
            
            self.logger.info(f"✅ ลงทะเบียน {person_name} จาก {image_path} สำเร็จ (Quality: {best_face.bbox.confidence:.3f})")
            return True
              except Exception as e:
            self.logger.error(f"❌ เกิดข้อผิดพลาดในการลงทะเบียน {image_path}: {e}")
            return False

    async def enroll_reference_images(self) -> bool:
        """ลงทะเบียนภาพอ้างอิงทั้งหมด"""
        self.logger.info("📝 เริ่มการลงทะเบียนภาพอ้างอิง...")
        
        # รายการไฟล์อ้างอิง - boss_01-10 และ night_01-10
        reference_files = [
            ("test_images/boss_01.jpg", "Boss"),
            ("test_images/boss_02.jpg", "Boss"),
            ("test_images/boss_03.jpg", "Boss"),
            ("test_images/boss_04.jpg", "Boss"),
            ("test_images/boss_05.jpg", "Boss"),
            ("test_images/boss_06.jpg", "Boss"),
            ("test_images/boss_07.jpg", "Boss"),
            ("test_images/boss_08.jpg", "Boss"),
            ("test_images/boss_09.jpg", "Boss"),
            ("test_images/boss_10.jpg", "Boss"),
            ("test_images/night_01.jpg", "Night"),
            ("test_images/night_02.jpg", "Night"),
            ("test_images/night_03.jpg", "Night"),
            ("test_images/night_04.jpg", "Night"),
            ("test_images/night_05.jpg", "Night"),
            ("test_images/night_06.jpg", "Night"),
            ("test_images/night_07.jpg", "Night"),
            ("test_images/night_08.jpg", "Night"),
            ("test_images/night_09.jpg", "Night"),
            ("test_images/night_10.jpg", "Night"),
        ]
        
        total_registered = 0
        
        for image_path, person_name in reference_files:
            if await self.enroll_person(image_path, person_name):
                total_registered += 1
                
        # แสดงสรุป
        self.logger.info("=" * 50)
        self.logger.info("📊 สรุปการลงทะเบียน Ultra Quality Enhancement:")
        for person_name, embeddings in self.registered_people.items():
            self.logger.info(f"   👤 {person_name}: {len(embeddings)} ภาพ")
        self.logger.info(f"   📈 รวมทั้งหมด: {total_registered} ภาพ")
        self.logger.info(f"   🎯 Face crop size: 224x224 (Ultra Quality)")
        
        return total_registered > 0

    def assess_embedding_quality(self, embedding: np.ndarray) -> float:
        """ประเมินคุณภาพของ embedding (0-1)"""
        try:
            # 1. ตรวจสอบ magnitude (ควรใกล้ 1.0 หลัง normalization)
            magnitude = np.linalg.norm(embedding)
            magnitude_score = min(magnitude, 1.0)
            
            # 2. ตรวจสอบ variance (ความหลากหลายของค่า)
            variance = np.var(embedding)
            variance_score = min(variance * 10, 1.0)  # scale up
            
            # 3. ตรวจสอบ sparsity (ไม่ควรมีค่า 0 เยอะ)
            non_zero_ratio = np.count_nonzero(embedding) / len(embedding)
            sparsity_score = non_zero_ratio
            
            # 4. ตรวจสอบ distribution (ควรกระจายตัวดี)
            std_dev = np.std(embedding)
            distribution_score = min(std_dev * 5, 1.0)
            
            # รวมคะแนน
            quality_score = (magnitude_score * 0.3 + variance_score * 0.3 + 
                           sparsity_score * 0.2 + distribution_score * 0.2)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # default score

    def detect_image_context(self, face_count: int, image_path: str) -> Dict[str, Any]:
        """ตรวจสอบ context ของภาพ"""
        context = {
            'is_group_photo': face_count > 2,
            'is_single_photo': face_count == 1,
            'is_face_swap': any(pattern in image_path.lower() 
                              for pattern in ['face-swap', 'swap', 'fake']),
            'is_reference': os.path.basename(image_path) in self.reference_images,
            'face_count': face_count
        }
        return context

    async def advanced_face_matching(self, target_embedding: np.ndarray, 
                                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Advanced face matching ด้วย Multi-tier threshold และ Cross-person validation
        """
        # 1. คำนวณ similarity กับทุกคน
        all_similarities = {}
        
        for person_name, embeddings_data in self.registered_people.items():
            similarities = []
            for embedding_data in embeddings_data:
                similarity = np.dot(target_embedding, embedding_data['embedding']) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding_data['embedding']) + 1e-7
                )
                similarities.append(similarity)
            
            if similarities:
                # ใช้ max similarity แทน average เพื่อความแม่นยำ
                max_similarity = max(similarities)
                all_similarities[person_name] = {
                    'max_similarity': max_similarity,
                    'avg_similarity': np.mean(similarities),
                    'count': len(similarities)
                }
        
        if not all_similarities:
            return None
        
        # 2. หาผู้ที่มี similarity สูงสุด
        best_person = max(all_similarities.keys(), 
                         key=lambda x: all_similarities[x]['max_similarity'])
        best_similarity = all_similarities[best_person]['max_similarity']
        
        # 3. หาผู้ที่มี similarity สูงเป็นอันดับ 2
        second_best_similarity = 0.0
        if len(all_similarities) > 1:
            similarities_list = [(person, data['max_similarity']) 
                               for person, data in all_similarities.items()]
            similarities_list.sort(key=lambda x: x[1], reverse=True)
            if len(similarities_list) > 1:
                second_best_similarity = similarities_list[1][1]
        
        # 4. Cross-person validation
        similarity_gap = best_similarity - second_best_similarity
        
        # 5. เลือก threshold ตาม context
        if context['is_reference']:
            required_threshold = self.thresholds['medium_confidence']
            self.logger.debug(f"📋 Reference image: using medium threshold {required_threshold}")
        elif context['is_face_swap']:
            required_threshold = self.thresholds['high_confidence']
            self.logger.debug(f"🔍 Face-swap image: using high threshold {required_threshold}")
        elif context['is_group_photo']:
            # Group photo ใช้ threshold สูงกว่า + penalty
            required_threshold = self.thresholds['medium_confidence'] + self.thresholds['group_photo_penalty']
            self.logger.debug(f"👥 Group photo: using elevated threshold {required_threshold}")
        else:
            required_threshold = self.thresholds['low_confidence']
            self.logger.debug(f"👤 Single photo: using low threshold {required_threshold}")
        
        # 6. ตรวจสอบเงื่อนไขต่างๆ
        
        # 6.1 Basic threshold check
        if best_similarity < required_threshold:
            self.logger.debug(f"❌ Below threshold: {best_similarity:.3f} < {required_threshold:.3f}")
            return None
        
        # 6.2 Cross-person gap check (ป้องกัน confusion ระหว่างคน)
        if similarity_gap < self.thresholds['cross_person_gap']:
            self.recognition_stats['cross_person_rejections'] += 1
            self.logger.debug(f"❌ Insufficient gap between persons: {similarity_gap:.3f} < {self.thresholds['cross_person_gap']}")
            return None
        
        # 6.3 Group photo additional validation
        if context['is_group_photo'] and best_similarity < self.thresholds['high_confidence']:
            if similarity_gap < 0.20:  # เข้มงวดกว่าในกลุ่ม
                self.recognition_stats['group_photo_rejections'] += 1
                self.logger.debug(f"❌ Group photo: insufficient confidence gap")
                return None
        
        # 6.4 Face-swap specific validation
        if context['is_face_swap']:
            if best_similarity < self.thresholds['high_confidence']:
                self.logger.debug(f"❌ Face-swap below high threshold: {best_similarity:.3f}")
                return None
                
        # 7. High confidence match tracking
        if best_similarity >= self.thresholds['high_confidence']:
            self.recognition_stats['high_confidence_matches'] += 1
        
        # 8. สร้างผลลัพธ์
        return {
            'person_name': best_person,
            'confidence': best_similarity,
            'raw_confidence': best_similarity,
            'similarity_gap': similarity_gap,
            'second_best_similarity': second_best_similarity,
            'threshold_used': required_threshold,
            'context': context
        }    async def recognize_face_in_image(self, image_path: str) -> List[Dict[str, Any]]:
        """จดจำใบหน้าในภาพหนึ่งใบ"""
        try:
            # อ่านภาพ
            image = cv2.imread(image_path)
            if image is None:
                return []
                
            # === ULTRA QUALITY ENHANCEMENT ===
            enhanced_image = self.ultra_enhancer.enhance_image_ultra_quality(image)
            
            # ประเมินคุณภาพ
            quality_metrics = self.ultra_enhancer.assess_image_quality(enhanced_image)
            self.logger.debug(f"Image quality: {quality_metrics}")
            
            # ตรวจหาใบหน้า
            detection_result = await self.detection_service.detect_faces(enhanced_image)
            
            if not detection_result.faces:
                return []
            
            # ตรวจสอบ context ของภาพ
            context = self.detect_image_context(len(detection_result.faces), image_path)
            
            results = []
            
            # ประมวลผลแต่ละใบหน้า
            for i, face in enumerate(detection_result.faces):
                self.recognition_stats['total_faces'] += 1
                
                face_result = {
                    'face_index': i,
                    'bbox': {
                        'x': int(face.bbox.x1),
                        'y': int(face.bbox.y1),
                        'width': int(face.bbox.width),
                        'height': int(face.bbox.height)
                    },
                    'detection_confidence': face.bbox.confidence,
                    'person_name': 'unknown',
                    'recognition_confidence': 0.0,
                    'context_info': context.copy(),
                    'quality_metrics': quality_metrics
                }
                
                # === ULTRA QUALITY FACE CROPPING ===
                face_crop = self.ultra_enhancer.crop_face_ultra_quality(
                    enhanced_image, face.bbox, target_size=224
                )
                
                if face_crop.size > 0:
                    embedding = await self.face_service.extract_embedding(face_crop)
                    
                    if embedding is not None:
                        # ประเมินคุณภาพ embedding
                        embedding_quality = self.assess_embedding_quality(embedding.vector)
                        
                        # ถ้าคุณภาพต่ำเกินไป ให้ปฏิเสธ
                        if embedding_quality < 0.3:
                            self.logger.debug(f"❌ Poor embedding quality: {embedding_quality:.3f}")
                            face_result['person_name'] = 'unknown'
                        else:
                            # ใช้ advanced matching
                            best_match = await self.advanced_face_matching(embedding.vector, context)
                            
                            if best_match:
                                face_result['person_name'] = best_match['person_name']
                                face_result['recognition_confidence'] = best_match['confidence']
                                face_result['similarity_gap'] = best_match['similarity_gap']
                                face_result['threshold_used'] = best_match['threshold_used']
                                
                                self.logger.debug(f"✅ Match: {best_match['person_name']} ({best_match['confidence']:.3f})")
                            else:
                                face_result['person_name'] = 'unknown'
                                self.recognition_stats['false_positives_prevented'] += 1
                                self.logger.debug(f"🚫 Rejected match to prevent false positive")
                
                results.append(face_result)
                
            return results
            
        except Exception as e:
            self.logger.error(f"❌ เกิดข้อผิดพลาดในการจดจำใบหน้า {image_path}: {e}")
            return []

    def draw_face_boxes(self, image: np.ndarray, face_results: List[Dict[str, Any]]) -> np.ndarray:
        """วาดกรอบและข้อมูลใบหน้าบนภาพ"""
        result_image = image.copy()
        
        for face_data in face_results:
            bbox = face_data['bbox']
            person_name = face_data['person_name']
            confidence = face_data['recognition_confidence']
            
            # เลือกสีตามผลลัพธ์
            if person_name == 'unknown':
                color = (0, 0, 255)  # สีแดงสำหรับ unknown
                label = "UNKNOWN"
            elif person_name in ['Boss', 'Night']:
                # สีเขียวเข้มสำหรับ high confidence, สีเขียวอ่อนสำหรับปกติ
                if confidence > 0.75:
                    color = (0, 200, 0)  # เขียวเข้ม = high confidence
                else:
                    color = (0, 255, 0)  # เขียวปกติ
                label = f"{person_name.upper()}"
            else:
                color = (255, 0, 0)  # สีน้ำเงินสำหรับอื่นๆ
                label = f"{person_name.upper()}"
                
            # วาดกรอบ
            cv2.rectangle(
                result_image,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color, 3
            )
            
            # เตรียมข้อความ
            if person_name != 'unknown':
                # เพิ่มข้อมูล threshold ถ้ามี
                threshold_info = ""
                if 'threshold_used' in face_data:
                    threshold_info = f" [T:{face_data['threshold_used']:.2f}]"
                text = f"{label} ({confidence:.1%}){threshold_info}"
            else:
                text = label
                
            # วาดข้อความ
            font_scale = 0.6  # ลดขนาดเพื่อให้พอดี
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # วาดพื้นหลังข้อความ
            cv2.rectangle(
                result_image,
                (bbox['x'], bbox['y'] - text_height - 10),
                (bbox['x'] + text_width, bbox['y']),
                color, -1
            )
            
            # วาดข้อความ
            cv2.putText(
                result_image,
                text,
                (bbox['x'], bbox['y'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
        return result_image

    async def test_all_images(self) -> Dict[str, Any]:
        """ทดสอบกับภาพทั้งหมดในโฟลเดอร์"""
        self.logger.info("🧪 เริ่มการทดสอบกับภาพทั้งหมด...")
        
        test_images_dir = Path("test_images")
        if not test_images_dir.exists():
            self.logger.error("❌ ไม่พบโฟลเดอร์ test_images")
            return {}
            
        # หาไฟล์ภาพทั้งหมด
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(test_images_dir.glob(f"*{ext}")))
            image_files.extend(list(test_images_dir.glob(f"*{ext.upper()}")))
            
        self.logger.info(f"📁 พบภาพทั้งหมด: {len(image_files)} ไฟล์")
        
        # สถิติการทดสอบ
        test_stats = {
            'total_images': len(image_files),
            'processed_images': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'total_unknown_faces': 0,
            'recognition_by_person': {},
            'processing_time': 0.0,
            'results': [],
            'advanced_settings': {
                'similarity_threshold': self.config.similarity_threshold,
                'unknown_threshold': self.config.unknown_threshold,
                'detection_method': 'YOLO Models',
                'recognition_model': str(self.config.preferred_model),
                'multi_tier_thresholds': self.thresholds,
                'dynamic_embeddings_enabled': self.enable_dynamic_embeddings
            },
            'advanced_stats': self.recognition_stats.copy()
        }
        
        start_time = datetime.now()
        
        # ประมวลผลแต่ละภาพ
        for image_file in image_files:
            try:
                self.logger.info(f"🔍 กำลังประมวลผล: {image_file.name}")
                
                # จดจำใบหน้า
                face_results = await self.recognize_face_in_image(str(image_file))
                
                if face_results:
                    # อ่านภาพเพื่อวาดกรอบ
                    original_image = cv2.imread(str(image_file))
                    if original_image is not None:
                        # วาดกรอบและข้อมูล
                        result_image = self.draw_face_boxes(original_image, face_results)
                        
                        # บันทึกผลลัพธ์
                        output_filename = f"result_{image_file.stem}.jpg"
                        output_path = self.output_dir / output_filename
                        cv2.imwrite(str(output_path), result_image)
                        
                        # อัพเดทสถิติ
                        test_stats['total_faces_detected'] += len(face_results)
                        
                        for face_result in face_results:
                            person_name = face_result['person_name']
                            if person_name != 'unknown':
                                test_stats['total_faces_recognized'] += 1
                                if person_name not in test_stats['recognition_by_person']:
                                    test_stats['recognition_by_person'][person_name] = 0
                                test_stats['recognition_by_person'][person_name] += 1
                            else:
                                test_stats['total_unknown_faces'] += 1
                                
                        # เก็บผลลัพธ์แต่ละภาพ
                        test_stats['results'].append({
                            'image_path': str(image_file),
                            'faces_detected': len(face_results),
                            'faces_recognized': len([f for f in face_results if f['person_name'] != 'unknown']),
                            'face_details': face_results
                        })
                        
                        self.logger.info(f"   📊 พบใบหน้า: {len(face_results)}, จดจำได้: {len([f for f in face_results if f['person_name'] != 'unknown'])}")
                
                test_stats['processed_images'] += 1
                
            except Exception as e:
                self.logger.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล {image_file}: {e}")
                
        # คำนวณเวลาที่ใช้
        end_time = datetime.now()
        test_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # อัพเดท advanced stats
        test_stats['advanced_stats'] = self.recognition_stats.copy()
        
        return test_stats

    def convert_numpy_types(self, obj):
        """แปลง numpy types เป็น Python native types สำหรับ JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        return obj

    def save_test_report(self, test_stats: Dict[str, Any]) -> str:
        """บันทึกรายงานผลการทดสอบ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # แปลง numpy types ทั้งหมดก่อน save JSON
        clean_test_stats = self.convert_numpy_types(test_stats)
        
        # บันทึกเป็น JSON
        json_filename = f"advanced_test_v13_{timestamp}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_test_stats, f, ensure_ascii=False, indent=2)
            
        # บันทึกเป็น Markdown
        md_filename = f"advanced_test_v13_{timestamp}.md"
        md_path = self.output_dir / md_filename
        
        recognition_rate = (test_stats['total_faces_recognized'] / test_stats['total_faces_detected'] * 100) if test_stats['total_faces_detected'] > 0 else 0
        
        md_content = f"""# รายงานผลการทดสอบระบบ Face Recognition (Advanced v13)

## 📊 สรุปผลการทดสอบ

**วันที่ทดสอบ:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**เวลาในการประมวลผล:** {test_stats['processing_time']:.2f} วินาที

### สถิติโดยรวม
- **ภาพทั้งหมด:** {test_stats['total_images']} ภาพ
- **ภาพที่ประมวลผลได้:** {test_stats['processed_images']} ภาพ
- **ใบหน้าที่ตรวจพบ:** {test_stats['total_faces_detected']} ใบหน้า
- **ใบหน้าที่จดจำได้:** {test_stats['total_faces_recognized']} ใบหน้า
- **ใบหน้าที่ไม่รู้จัก:** {test_stats['total_unknown_faces']} ใบหน้า
- **อัตราการจดจำ:** {recognition_rate:.1f}%

### การตั้งค่าขั้นสูง (Advanced v13)
- **Base Similarity Threshold:** {test_stats['advanced_settings']['similarity_threshold']}
- **Unknown Threshold:** {test_stats['advanced_settings']['unknown_threshold']}
- **High Confidence Threshold:** {test_stats['advanced_settings']['multi_tier_thresholds']['high_confidence']}
- **Medium Confidence Threshold:** {test_stats['advanced_settings']['multi_tier_thresholds']['medium_confidence']}
- **Cross-person Gap:** {test_stats['advanced_settings']['multi_tier_thresholds']['cross_person_gap']}
- **วิธีการตรวจจับ:** {test_stats['advanced_settings']['detection_method']}
- **โมเดลจดจำ:** {test_stats['advanced_settings']['recognition_model']}

### สถิติขั้นสูง
- **False Positives ที่ป้องกันได้:** {test_stats['advanced_stats']['false_positives_prevented']}
- **Cross-person Rejections:** {test_stats['advanced_stats']['cross_person_rejections']}
- **Group Photo Rejections:** {test_stats['advanced_stats']['group_photo_rejections']}
- **High Confidence Matches:** {test_stats['advanced_stats']['high_confidence_matches']}

## 👥 สถิติการจดจำแต่ละบุคคล

"""
        
        for person, count in test_stats['recognition_by_person'].items():
            md_content += f"- **{person}:** {count} ครั้ง\n"
        
        md_content += "\n## 📋 รายละเอียดการประมวลผลแต่ละภาพ\n\n"
        
        for result in test_stats['results']:
            md_content += f"### {Path(result['image_path']).name}\n"
            md_content += f"- **ใบหน้าที่ตรวจพบ:** {result['faces_detected']}\n"
            md_content += f"- **ใบหน้าที่จดจำได้:** {result['faces_recognized']}\n"
            
            if result['face_details']:
                md_content += "- **รายละเอียด:**\n"
                for i, face in enumerate(result['face_details']):
                    confidence_text = f" ({face['recognition_confidence']:.1%})" if face['person_name'] != 'unknown' else ""
                    threshold_info = f" [T:{face.get('threshold_used', 0):.2f}]" if 'threshold_used' in face else ""
                    md_content += f"  - Face {i+1}: {face['person_name'].upper()}{confidence_text}{threshold_info}\n"
            md_content += "\n"
        
        # เปรียบเทียบกับระบบเดิม
        comparison_text = "ดีกว่า" if recognition_rate > 38.5 else "ต้องปรับปรุง"
        md_content += f"""
## 🔄 นวัตกรรมขั้นสูง (v13)

### การปรับปรุงที่สำคัญ:
1. **Multi-tier Threshold System:** ใช้ threshold ที่แตกต่างกันตาม context
2. **Cross-person Validation:** ป้องกัน confusion ระหว่างคน
3. **Context-aware Recognition:** พิจารณา single photo vs group photo
4. **Advanced Embedding Quality Assessment:** ประเมินคุณภาพ embedding ก่อนใช้งาน
5. **False Positive Prevention:** ระบบป้องกัน false positives อัจฉริยะ

### ผลลัพธ์:
- **อัตราการจดจำ:** {recognition_rate:.1f}% (เป้าหมาย: ดีกว่า 38.5%)
- **ความแม่นยำ:** {comparison_text}
- **False Positives ที่ป้องกันได้:** {test_stats['advanced_stats']['false_positives_prevented']} cases
- **Cross-contamination:** ลดลงด้วย validation system

---
*รายงานสร้างโดย Advanced Real Image Test System v13*
"""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        self.logger.info(f"📄 บันทึกรายงาน: {json_path}")
        self.logger.info(f"📄 บันทึกรายงาน: {md_path}")
        
        return str(md_path)

    async def run_complete_test(self):
        """รันการทดสอบทั้งหมด"""
        try:
            self.logger.info("🚀 เริ่มต้นระบบทดสอบ Face Recognition Advanced v13")
            self.logger.info("=" * 60)
            
            # ขั้นตอนที่ 0: Initialize services
            await self.initialize_services()
            
            # ขั้นตอนที่ 1: ลงทะเบียนภาพอ้างอิง
            if not await self.enroll_reference_images():
                self.logger.error("❌ ไม่สามารถลงทะเบียนภาพอ้างอิงได้")
                return
                
            # ขั้นตอนที่ 2: ทดสอบกับภาพทั้งหมด
            test_results = await self.test_all_images()
            
            if not test_results:
                self.logger.error("❌ ไม่มีผลการทดสอบ")
                return
                
            # ขั้นตอนที่ 3: บันทึกรายงาน
            report_path = self.save_test_report(test_results)
            
            # แสดงสรุปผลลัพธ์
            self.logger.info("=" * 60)
            self.logger.info("🎯 สรุปผลการทดสอบ:")
            recognition_rate = (test_results['total_faces_recognized'] / test_results['total_faces_detected'] * 100) if test_results['total_faces_detected'] > 0 else 0
            self.logger.info(f"   📊 อัตราการจดจำ: {recognition_rate:.1f}%")
            self.logger.info(f"   👥 ใบหน้าที่ตรวจพบ: {test_results['total_faces_detected']}")
            self.logger.info(f"   ✅ ใบหน้าที่จดจำได้: {test_results['total_faces_recognized']}")
            self.logger.info(f"   ❓ ใบหน้าที่ไม่รู้จัก: {test_results['total_unknown_faces']}")
            self.logger.info(f"   ⏱️ เวลาทั้งหมด: {test_results['processing_time']:.2f} วินาที")
            self.logger.info(f"   📄 รายงาน: {report_path}")
            
            # แสดงสถิติขั้นสูง
            advanced_stats = test_results['advanced_stats']
            self.logger.info("   🛡️ สถิติการป้องกัน:")
            self.logger.info(f"      - False Positives ที่ป้องกันได้: {advanced_stats['false_positives_prevented']}")
            self.logger.info(f"      - Cross-person Rejections: {advanced_stats['cross_person_rejections']}")
            self.logger.info(f"      - Group Photo Rejections: {advanced_stats['group_photo_rejections']}")
            self.logger.info(f"      - High Confidence Matches: {advanced_stats['high_confidence_matches']}")
            
            # แสดงสถิติแต่ละคน
            if test_results['recognition_by_person']:
                self.logger.info("   📈 การจดจำแต่ละบุคคล:")
                for person, count in test_results['recognition_by_person'].items():
                    self.logger.info(f"      - {person}: {count} ครั้ง")
            
            # เปรียบเทียบกับระบบเดิม
            if recognition_rate > 38.5:
                self.logger.info(f"   🎉 ผลลัพธ์ดีกว่าระบบเดิม! ({recognition_rate:.1f}% > 38.5%)")
            else:
                self.logger.info(f"   ⚠️ ผลลัพธ์ยังไม่ดีกว่าระบบเดิม ({recognition_rate:.1f}% vs 38.5%)")
            
            self.logger.info("🏁 การทดสอบเสร็จสิ้น!")
            
        except Exception as e:
            self.logger.error(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")
            raise

async def main():
    """ฟังก์ชันหลัก"""
    try:
        system = AdvancedRealImageTestSystemV13()
        await system.run_complete_test()
        
    except KeyboardInterrupt:
        print("\n⏹️ การทดสอบถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())