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
from .utils import BoundingBox, FaceDetection, DetectionResult, calculate_face_quality # Import calculate_face_quality from .utils

# Setup logging
logger = logging.getLogger(__name__)

class EnhancedDetectorAdapter:
    """
    อะแดปเตอร์สำหรับเชื่อมต่อ EnhancedFaceDetector เข้ากับ FaceDetectionService
    เพื่อรองรับภาพที่มีความท้าทาย เช่น ภาพกลางคืน ภาพกลุ่ม และ ภาพ face-swap
    """
    
    def __init__(self, 
                 vram_manager=None, 
                 use_cuda: bool = True,
                 config: Optional[Dict[str, Any]] = None): # Added config
        """
        กำหนดค่าเริ่มต้นสำหรับอะแดปเตอร์
        
        Args:
            vram_manager: ตัวจัดการ VRAM (ใช้ตาม interface เดิม แต่ไม่ได้ใช้งานจริง)
            use_cuda: ใช้ GPU ในการประมวลผลหรือไม่
            config: Configuration dictionary for the detector (e.g., model paths)
        """
        self.vram_manager = vram_manager
        self.use_cuda = use_cuda
        self.config = config if config is not None else {} # Store config
        self.enhanced_detector = None
        self.model_loaded = False
        self.model_name = "enhanced_detector_v2" # Added model_name attribute

    async def initialize(self) -> bool:
        """เริ่มต้นการทำงานของ Enhanced Face Detector"""
        try:
            # Import EnhancedFaceDetector - assuming it's at the project root (d:\\projec-final)
            # and the script is run from there, making it directly importable.
            from enhanced_face_detector import EnhancedFaceDetector
            
            # Create enhanced detector instance, passing config if it expects it
            self.enhanced_detector = EnhancedFaceDetector(
                use_cuda=self.use_cuda, 
                config=self.config  # Pass the adapter's config
            )
            
            # Load detector models
            load_success = await self.enhanced_detector.load_models() # Assuming load_models is async
            self.model_loaded = load_success
            
            if load_success:
                logger.info(f"Enhanced Face Detector ({self.model_name}) initialized successfully")
                return True
            else:
                logger.error(f"Failed to load Enhanced Face Detector ({self.model_name}) models")
                return False
                
        except ImportError as e:
            logger.error(f"Could not import EnhancedFaceDetector. Ensure 'enhanced_face_detector.py' is in the project root (d:\\projec-final) and the script is run from there. Error: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error initializing Enhanced Face Detector ({self.model_name}): {e}", exc_info=True)
            return False

    async def detect_faces_raw(self, 
                             image: np.ndarray, 
                             conf_threshold: Optional[float] = 0.5, 
                             iou_threshold: Optional[float] = 0.45) -> Tuple[List[np.ndarray], float]:
        """
        ตรวจจับใบหน้าและส่งคืน raw detection results (bboxes, scores, class_ids) และเวลาประมวลผลของโมเดล
        """
        if not self.model_loaded or self.enhanced_detector is None:
            logger.warning(f"Enhanced Face Detector ({self.model_name}) not initialized for detect_faces_raw")
            return [], 0.0

        model_process_start_time = time.time()
        try:
            # Assuming enhanced_detector has a method that returns raw-like detections
            # This might be `detect_faces` itself if it returns a compatible format,
            # or a new/different method.
            # The original `detect_faces` in the adapter returned: success, detections, result_image
            # We need a list of [x1, y1, x2, y2, score, class_id (optional)]
            
            # Let's assume self.enhanced_detector.detect_raw() or similar exists
            # or adapt self.enhanced_detector.detect_faces()
            
            # If EnhancedFaceDetector's detect_faces returns (success, detections, result_image)
            # where detections are [[x1,y1,x2,y2,conf], ...]
            
            # Apply thresholds if the underlying detector doesn't handle them internally via these params
            # This depends on EnhancedFaceDetector's implementation.
            # For now, assume they are passed or handled if necessary.

            raw_detections_from_model = await self.enhanced_detector.detect_raw_for_adapter(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold 
            ) # This method needs to exist in EnhancedFaceDetector
              # It should return a list of arrays: [[x1,y1,x2,y2,score, optional_class_id], ...]

            model_processing_time = time.time() - model_process_start_time
            return raw_detections_from_model, model_processing_time

        except Exception as e:
            logger.error(f"Error in detect_faces_raw with {self.model_name}: {e}", exc_info=True)
            model_processing_time = time.time() - model_process_start_time
            return [], model_processing_time
    
    async def detect_faces(self, 
                         image: np.ndarray,
                         confidence_threshold: Optional[float] = 0.5, # Renamed from conf_threshold
                         iou_threshold: Optional[float] = 0.45,    # Default from your utils.py
                         max_faces: Optional[int] = None,
                         min_face_size: Optional[int] = None, # Changed from Tuple[int,int] to int
                         # model_name, return_landmarks are no longer top-level params here
                         # as per the new FaceDetectionService.detect_faces signature
                         ) -> DetectionResult:
        """
        ตรวจจับใบหน้าโดยใช้ EnhancedDetectorAdapter, สอดคล้องกับ FaceDetectionService.detect_faces ใหม่
        """
        overall_start_time = time.time()
        
        if not self.model_loaded or self.enhanced_detector is None:
            logger.warning(f"Enhanced Face Detector ({self.model_name}) not initialized.")
            return DetectionResult(
                faces=[],
                image_shape=image.shape[:2], # image_shape is Tuple[int, int]
                total_processing_time=time.time() - overall_start_time,
                model_used=self.model_name,
                error="Detector not initialized"
            )
        
        try:
            # Use detect_faces_raw to get raw detections and model_processing_time
            raw_detections, model_processing_time = await self.detect_faces_raw(
                image, 
                conf_threshold=confidence_threshold, 
                iou_threshold=iou_threshold
            )

            processed_faces: List[FaceDetection] = []
            if raw_detections:
                for det_array in raw_detections:
                    if len(det_array) < 5:
                        logger.warning(f"Skipping malformed detection array: {det_array}")
                        continue
                    
                    x1, y1, x2, y2, score = det_array[:5]
                    class_id = int(det_array[5]) if len(det_array) > 5 else None # Optional class_id

                    bbox = BoundingBox(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                        confidence=float(score),
                        class_id=class_id 
                    )

                    # Validate bounding box (optional, but good practice)
                    # from .utils import validate_bounding_box # Ensure this is imported
                    # if not validate_bounding_box(bbox, image.shape[:2]):
                    #     logger.debug(f"Skipping invalid bbox: {bbox}")
                    #     continue

                    # Filter by min_face_size (pixel area)
                    if min_face_size is not None and bbox.area < (min_face_size * min_face_size): # Assuming min_face_size is width
                         logger.debug(f"Skipping small face: area {bbox.area} < {min_face_size*min_face_size}")
                         continue

                    # Calculate face quality (using the imported function)
                    # Ensure face_roi is correctly cropped for quality calculation
                    face_roi = image[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]
                    if face_roi.size == 0: # Check if ROI is valid
                        quality_score = 0.0
                    else:
                        # Ensure calculate_face_quality is correctly imported and used
                        quality_score = calculate_face_quality(face_roi, bbox) # Pass bbox as well

                    face_detection_obj = FaceDetection(
                        bbox=bbox,
                        quality_score=quality_score,
                        model_used=self.model_name, # This specific adapter/model
                        processing_time=model_processing_time / len(raw_detections) if raw_detections else 0, # Approximate per face
                        # landmarks are not handled by this adapter's current raw output
                    )
                    processed_faces.append(face_detection_obj)

            # Sort by confidence (or quality_score) and limit by max_faces
            if processed_faces:
                processed_faces.sort(key=lambda f: f.bbox.confidence, reverse=True) # Or f.quality_score
                if max_faces is not None and len(processed_faces) > max_faces:
                    processed_faces = processed_faces[:max_faces]
            
            total_processing_time = time.time() - overall_start_time
            
            return DetectionResult(
                faces=processed_faces,
                image_shape=image.shape[:2], # Corrected to Tuple[int, int]
                total_processing_time=total_processing_time,
                model_used=self.model_name, # Overall model for this result
                # fallback_used is not applicable here unless enhanced_detector has such a concept
            )
            
        except Exception as e:
            logger.error(f"Error during face detection with {self.model_name}: {e}", exc_info=True)
            total_processing_time = time.time() - overall_start_time
            return DetectionResult(
                faces=[],
                image_shape=image.shape[:2], # Corrected
                total_processing_time=total_processing_time,
                model_used=self.model_name,
                error=str(e)
            )
    
    async def cleanup(self) -> bool:
        """Clean up resources"""
        try:
            if self.enhanced_detector and hasattr(self.enhanced_detector, 'cleanup'):
                await self.enhanced_detector.cleanup() # Assuming async cleanup
            self.enhanced_detector = None
            self.model_loaded = False
            logger.info(f"Enhanced Face Detector ({self.model_name}) cleaned up successfully.")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up Enhanced Face Detector ({self.model_name}): {e}", exc_info=True)
            return False

# Example of how EnhancedFaceDetector might need to be structured or imported:
# This is illustrative. Your actual EnhancedFaceDetector class will be elsewhere.
#
# class EnhancedFaceDetector:
#     def __init__(self, use_cuda: bool = True, config: Optional[Dict[str, Any]] = None):
#         self.use_cuda = use_cuda
#         self.config = config
#         self.min_confidence = 0.5 # Default
#         self.min_face_size = (30,30) # Default
#         # ... other initializations ...
#
#     async def load_models(self):
#         # ... model loading logic ...
#         print(f"EnhancedFaceDetector models loaded with config: {self.config}")
#         return True
#
#     async def detect_raw_for_adapter(self, image, conf_threshold, iou_threshold):
#         # This method should perform detection and return a list of numpy arrays
#         # Each array: [x1, y1, x2, y2, score, class_id (optional)]
#         # Example:
#         # detections = [[10,10,100,100,0.9,0], [120,30,200,150,0.85,0]]
#         # return [np.array(d) for d in detections]
#         print(f"EnhancedFaceDetector detect_raw_for_adapter called with conf: {conf_threshold}, iou: {iou_threshold}")
#         # Placeholder:
#         if image is not None:
#             return [np.array([10.0, 10.0, 110.0, 110.0, 0.95, 0]),
#                     np.array([50.0, 50.0, 150.0, 150.0, 0.88, 0])], 0.1
#         return [], 0.0
#
#     async def cleanup(self):
#         print("EnhancedFaceDetector cleaned up.")
#         return True
