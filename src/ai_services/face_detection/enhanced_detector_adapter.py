# cSpell:disable
"""
Enhanced Face Detector Adapter
อะแดปเตอร์สำหรับเชื่อมต่อ EnhancedFaceDetector เข้ากับ FaceDetectionService
"""
import os
import sys
import cv2
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Import necessary modules
from .utils import BoundingBox, FaceDetection, DetectionResult

# Setup logging
logger = logging.getLogger(__name__)

class EnhancedDetectorAdapter:
    """
    อะแดปเตอร์สำหรับเชื่อมต่อ EnhancedFaceDetector เข้ากับ FaceDetectionService
    เพื่อรองรับภาพที่มีความท้าทาย เช่น ภาพกลางคืน ภาพกลุ่ม และ ภาพ face-swap
    """
    
    def __init__(self, vram_manager=None, use_cuda: bool = True):
        """
        กำหนดค่าเริ่มต้นสำหรับอะแดปเตอร์
        
        Args:
            vram_manager: ตัวจัดการ VRAM (ใช้ตาม interface เดิม แต่ไม่ได้ใช้งานจริง)
            use_cuda: ใช้ GPU ในการประมวลผลหรือไม่
        """
        self.vram_manager = vram_manager
        self.use_cuda = use_cuda
        self.enhanced_detector = None
        self.model_loaded = False

    async def initialize(self) -> bool:
        """เริ่มต้นการทำงานของ Enhanced Face Detector"""
        try:
            # Import here to avoid circular imports
            from enhanced_face_detector import EnhancedFaceDetector
            
            # Create enhanced detector instance
            self.enhanced_detector = EnhancedFaceDetector(use_cuda=self.use_cuda)
            
            # Load detector models
            load_success = self.enhanced_detector.load_models()
            self.model_loaded = load_success
            
            if load_success:
                logger.info("Enhanced Face Detector initialized successfully")
                return True
            else:
                logger.error("Failed to load Enhanced Face Detector models")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Enhanced Face Detector: {e}")
            return False
    
    async def detect_faces(self, 
                         image: np.ndarray,
                         model_name: Optional[str] = None,
                         conf_threshold: Optional[float] = None,
                         iou_threshold: Optional[float] = None,
                         return_landmarks: bool = False,
                         min_face_size: Optional[Tuple[int, int]] = None,
                         max_faces: Optional[int] = None) -> DetectionResult:
        """
        ตรวจจับใบหน้าด้วยวิธี Ensemble โดยอิงตาม interface เดิม
        
        Args:
            image: รูปภาพที่ต้องการตรวจจับในรูปแบบ numpy array
            model_name: ชื่อโมเดลที่ต้องการใช้ (ไม่ได้ใช้ แต่เก็บไว้เพื่อความเข้ากันได้)
            conf_threshold: ค่า confidence threshold
            iou_threshold: ค่า IOU threshold สำหรับ NMS
            return_landmarks: ต้องการ landmarks หรือไม่
            min_face_size: ขนาดใบหน้าขั้นต่ำที่จะตรวจจับ
            max_faces: จำนวนใบหน้าสูงสุดที่จะส่งคืน
            
        Returns:
            DetectionResult: ผลลัพธ์การตรวจจับใบหน้า
        """
        start_time = time.time()
        
        if not self.model_loaded:
            logger.warning("Enhanced Face Detector not initialized")
            # Return empty result
            return DetectionResult(
                faces=[],
                image_shape=image.shape,
                total_processing_time=0.0,
                model_used="enhanced_detector",
                error="Detector not initialized"
            )
        
        try:
            # Apply confidence threshold if provided
            if conf_threshold is not None:
                self.enhanced_detector.min_confidence = conf_threshold
            
            # Apply min face size if provided
            if min_face_size is not None:
                self.enhanced_detector.min_face_size = min_face_size
            
            # Call the enhanced detector
            success, detections, result_image = self.enhanced_detector.detect_faces(image)
            
            # If detection failed, return empty result
            if not success or not detections:
                total_time = time.time() - start_time
                return DetectionResult(
                    faces=[],
                    image_shape=image.shape,
                    total_processing_time=total_time,
                    model_used="enhanced_detector",
                    error="No faces detected"
                )
            
            # Convert detections to the FaceDetection format
            face_detections = []
            for i, det in enumerate(detections):
                # Extract bbox coordinates and confidence
                x1, y1, x2, y2, conf = map(float, det[:5])
                
                # Create BoundingBox object
                bbox = BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf
                )
                  # Create face detection object
                face_detection = FaceDetection(
                    bbox=bbox,
                    model_used="enhanced_detector"
                )
                
                face_detections.append(face_detection)
            
            # Limit number of faces if max_faces is specified
            if max_faces is not None and len(face_detections) > max_faces:
                # Sort by confidence and take top max_faces
                face_detections.sort(key=lambda f: f.bbox.confidence, reverse=True)
                face_detections = face_detections[:max_faces]
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Return detection result
            return DetectionResult(
                faces=face_detections,
                image_shape=image.shape,
                total_processing_time=total_time,
                model_used="enhanced_detector"
            )
            
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            total_time = time.time() - start_time
            return DetectionResult(
                faces=[],
                image_shape=image.shape,
                total_processing_time=total_time,
                model_used="enhanced_detector",
                error=str(e)
            )
    
    async def cleanup(self) -> bool:
        """Clean up resources"""
        try:
            # Clean up detector resources if any
            self.enhanced_detector = None
            self.model_loaded = False
            return True
        except Exception as e:
            logger.error(f"Error cleaning up Enhanced Face Detector: {e}")
            return False
