# cSpell:disable
# mypy: ignore-errors
"""
บริการตรวจจับใบหน้าอัจฉริยะ (Enhanced Intelligent Face Detection Service) ที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m
ระบบใช้ 4 ขั้นตอนอัจฉริยะในการตัดสินใจเลือกโมเดลที่เหมาะสมที่สุด และวิเคราะห์คุณภาพใบหน้า
"""
import time
import logging
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
import asyncio # Add asyncio import

from .yolo_models import YOLOv9ONNXDetector, YOLOv11Detector
from .utils import BoundingBox, FaceDetection, DetectionResult, calculate_face_quality, validate_bounding_box # filter_detection_results removed, validate_bounding_box added
# Import Enhanced Detector Adapter
from .enhanced_detector_adapter import EnhancedDetectorAdapter

# ไม่ import VRAMManager แต่สร้าง stub class ขึ้นมาแทน
class VRAMManager:
    """Stub class for VRAMManager."""
    async def request_model_allocation(self, *args, **kwargs):
        class Allocation:
            class Location:
                value = "cpu"
            location = Location()
        return Allocation()
    
    async def release_model_allocation(self, *args, **kwargs):
        return True
        
    async def get_vram_status(self, *args, **kwargs):
        return {"status": "stub", "available": 0}

logger = logging.getLogger(__name__)


# FIXED VERSION: Add fallback_opencv_detection function
def fallback_opencv_detection(image: np.ndarray,
                              scale_factor: float = 1.1,
                              min_neighbors: int = 5,
                              min_size: Tuple[int, int] = (30, 30)) -> List[BoundingBox]:
    """Detects faces using OpenCV Haar Cascade as a fallback."""
    try:
        # Convert to grayscale if needed
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image
        
        # Load Haar Cascade classifier
        # Ensure the path to the Haar Cascade XML file is correct
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            logger.error(f"Haar Cascade file not found at {cascade_path}")
            return []
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
        
        bboxes = []
        for (x, y, w, h) in faces:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            # Default confidence for Haar, class_id is removed from BoundingBox constructor call
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.5)
            bboxes.append(bbox)
        return bboxes
    except Exception as e:
        logger.error(f"Error in fallback_opencv_detection: {e}")
        return []

# FIXED VERSION: Add get_relaxed_face_detection_config
def get_relaxed_face_detection_config() -> Dict[str, Any]:
    """
    Provides a relaxed configuration for face detection, suitable for scenarios
    where maximizing detection recall is prioritized, even at the cost of precision
    or detection quality stringency.
    Values aligned with "🔧 คู่มือแก้ไขปัญหา 'จับหน้าไม่ได้' แบบทันที".
    """
    return {
        # General service settings
        'use_enhanced_detector': False, 

        # Model paths
        'yolov9c_model_path': 'model/face-detection/yolov9c-face-lindevs.onnx',
        'yolov9e_model_path': 'model/face-detection/yolov9e-face-lindevs.onnx',
        'yolov11m_model_path': 'model/face-detection/yolov11m-face.pt',

        # Decision criteria for model selection (relaxed)
        'max_usable_faces_yolov9': 12,  # From guide
        'min_agreement_ratio': 0.5,   # From guide
        'min_quality_threshold': 40,  # From guide
        'iou_threshold_agreement': 0.3, 

        # Detection parameters (relaxed)
        'conf_threshold': 0.10,       # From guide
        'iou_threshold_nms': 0.35,    
        'img_size': 640,             

        # FaceQualityAnalyzer configuration (relaxed - values from utils.py RELAXED VERSION)
        'quality_config': {
            'min_quality_threshold': 40, # Match guide
            'size_weight': 30,          
            'area_weight': 25,          
            'confidence_weight': 30,    
            'aspect_weight': 15,        
            'excellent_size': (80, 80), 
            'good_size': (50, 50),      
            'acceptable_size': (24, 24),
            'minimum_size': (8, 8), # from utils.py relaxed
            'bonus_score_for_high_confidence': 5.0, # from utils.py relaxed
            'high_confidence_threshold': 0.7 # from utils.py relaxed
        },

        # Fallback strategy configuration
        'fallback_config': {
            'enable_fallback_system': True,
            'max_fallback_attempts': 3, 
            'fallback_models': [ 
                {'model_name': 'yolov11m', 'conf_threshold': 0.15, 'iou_threshold': 0.35, 'min_faces_to_accept': 1},
                {'model_name': 'yolov9c', 'conf_threshold': 0.05, 'iou_threshold': 0.3, 'min_faces_to_accept': 1}, 
                {'model_name': 'opencv_haar', 'scale_factor': 1.1, 'min_neighbors': 3, 'min_size': (20,20), 'min_faces_to_accept': 1}
            ],
            'min_detections_after_fallback': 1, 
            'always_run_all_fallbacks_if_zero_initial': True, 
        },
        
        # Filter settings for _create_result (using relaxed values from utils.py)
        'filter_min_quality': 30.0, # This aligns with the relaxed utils.py, but guide suggests 40 for overall.
                                    # Let's use the guide's min_quality_threshold for filtering as well for consistency.
        'filter_min_quality_final': 40.0 # From guide for final filtering
    }

class QualityCategory(Enum):
    """ระดับคุณภาพของใบหน้า"""
    EXCELLENT = "excellent"  # คุณภาพดีเยี่ยม (80-100)
    GOOD = "good"            # คุณภาพดี (70-79)
    ACCEPTABLE = "acceptable"  # คุณภาพพอใช้ (min_quality_threshold-69 or 79 if higher)
    POOR = "poor"            # คุณภาพต่ำ (<min_quality_threshold)


class FaceQualityAnalyzer:
    """ระบบวิเคราะห์คุณภาพใบหน้า"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        ตั้งค่าเริ่มต้นสำหรับระบบวิเคราะห์คุณภาพใบหน้า
        
        Args:
            config: การตั้งค่าสำหรับการวิเคราะห์คุณภาพ
        """
        self.quality_weights = {
            'size_weight': config.get('size_weight', 30),
            'area_weight': config.get('area_weight', 25),
            'confidence_weight': config.get('confidence_weight', 30),
            'aspect_weight': config.get('aspect_weight', 15)
        }
        
        self.size_thresholds = {
            'excellent': config.get('excellent_size', (80, 80)),
            'good': config.get('good_size', (50, 50)),
            'acceptable': config.get('acceptable_size', (24, 24)),
            'minimum': config.get('minimum_size', (8, 8)) # from utils.py relaxed
        }
        
        self.min_quality_threshold = config.get('min_quality_threshold', 40) # from guide
        self.bonus_score_for_high_confidence = config.get('bonus_score_for_high_confidence', 5.0) # from utils.py relaxed
        self.high_confidence_threshold = config.get('high_confidence_threshold', 0.7) # from utils.py relaxed
    
    def get_quality_category(self, score: float) -> QualityCategory:
        """ระบุระดับคุณภาพตามคะแนน"""
        if score >= 80:
            return QualityCategory.EXCELLENT
        elif score >= 70: # Assuming good starts at 70, adjust if needed
            return QualityCategory.GOOD
        elif score >= self.min_quality_threshold: # Acceptable is now based on the dynamic threshold
            return QualityCategory.ACCEPTABLE
        else:
            return QualityCategory.POOR
    
    def is_face_usable(self, face: FaceDetection) -> bool:
        """ตรวจสอบว่าใบหน้าใช้งานได้หรือไม่"""
        if face.quality_score is None:
            return False
        return face.quality_score >= self.min_quality_threshold
    
    def analyze_detection_quality(self, faces: List[FaceDetection]) -> Dict[str, Any]:
        """
        วิเคราะห์คุณภาพของการตรวจจับใบหน้าทั้งหมด
        
        Args:
            faces: รายการใบหน้าที่ตรวจพบ
            
        Returns:
            ข้อมูลคุณภาพการตรวจจับ
        """
        if not faces:
            return {
                'total_count': 0,
                'usable_count': 0,
                'quality_ratio': 0.0,
                'quality_categories': {
                    QualityCategory.EXCELLENT.value: 0,
                    QualityCategory.GOOD.value: 0,
                    QualityCategory.ACCEPTABLE.value: 0,
                    QualityCategory.POOR.value: 0
                },
                'avg_quality': 0.0
            }
        
        quality_categories = {
            QualityCategory.EXCELLENT.value: 0,
            QualityCategory.GOOD.value: 0,
            QualityCategory.ACCEPTABLE.value: 0,
            QualityCategory.POOR.value: 0
        }
        
        usable_count = 0
        quality_scores = []
        
        for face in faces:
            if face.quality_score is not None:
                quality_scores.append(face.quality_score)
                category = self.get_quality_category(face.quality_score)
                quality_categories[category.value] += 1
                
                if self.is_face_usable(face):
                    usable_count += 1
        
        total_count = len(faces)
        quality_ratio = (usable_count / total_count) * 100 if total_count > 0 else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'total_count': total_count,
            'usable_count': usable_count,
            'quality_ratio': quality_ratio,
            'quality_categories': quality_categories,
            'avg_quality': avg_quality
        }

class DecisionResult:
    """ผลลัพธ์การตัดสินใจเลือกโมเดล"""
    
    def __init__(self):
        # ผลลัพธ์ขั้นตอนที่ 1: ทดสอบ YOLOv9
        self.yolov9c_detections = []
        self.yolov9e_detections = []
        self.yolov9c_time = 0.0
        self.yolov9e_time = 0.0
        
        # ผลลัพธ์ขั้นตอนที่ 2: การวิเคราะห์ความเห็นด้วย
        self.agreement = False
        self.agreement_ratio = 0.0
        self.agreement_type = ""
        
        # ผลลัพธ์ขั้นตอนที่ 3: การตัดสินใจ
        self.use_yolov11m = False
        self.decision_reasons = []
        
        # ผลลัพธ์ขั้นตอนที่ 4: ผลลัพธ์สุดท้าย
        self.final_detections = []
        self.final_model = ""
        self.final_time = 0.0
        
        # ข้อมูลคุณภาพ
        self.quality_info = {}
        
        # เวลาที่ใช้ทั้งหมด
        self.total_time = 0.0

        # Fallback information
        self.fallback_attempts_info = [] # List of dicts detailing each fallback attempt
        self.fallback_used = False
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON"""
        return {
            'step1_results': {
                'yolov9c': {
                    'count': len(self.yolov9c_detections),
                    'time': self.yolov9c_time
                },
                'yolov9e': {
                    'count': len(self.yolov9e_detections),
                    'time': self.yolov9e_time
                }
            },
            'step2_agreement': {
                'agreement': self.agreement,
                'ratio': self.agreement_ratio,
                'type': self.agreement_type
            },
            'step3_decision': {
                'use_yolov11m': self.use_yolov11m,
                'reasons': self.decision_reasons
            },
            'step4_results': {
                'model_used': self.final_model,
                'count': len(self.final_detections),
                'time': self.final_time
            },
            'quality_info': self.quality_info,
            'total_time': self.total_time,
            'fallback_info': { # Added fallback info
                'fallback_used': self.fallback_used,
                'attempts': self.fallback_attempts_info
            }
        }

class FaceDetectionService:
    """
    บริการตรวจจับใบหน้าอัจฉริยะที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m
    """
    def __init__(self, vram_manager: VRAMManager, config: Optional[Dict[str, Any]] = None): # config can be None
        """
        ตั้งค่าเริ่มต้นสำหรับบริการตรวจจับใบหน้า
        
        Args:
            vram_manager: ตัวจัดการหน่วยความจำ GPU
            config: การตั้งค่าสำหรับบริการ. If None, uses get_relaxed_face_detection_config().        """
        self.vram_manager = vram_manager
        self.logger = logging.getLogger(__name__)  # Add missing logger
        # ENHANCED VERSION: Use get_relaxed_face_detection_config if no config is provided or for specific keys
        self.config = config if config is not None else get_relaxed_face_detection_config()

        self.models: dict[str, Union[YOLOv9ONNXDetector, YOLOv11Detector, EnhancedDetectorAdapter]] = {} # Allow EnhancedDetectorAdapter
        self.model_stats: dict[str, dict[str, Union[float, int]]] = {}
        
        self.use_enhanced_detector = self.config.get('use_enhanced_detector', False)
        
        # ENHANCED VERSION: Use relaxed decision_criteria from config
        self.decision_criteria = {
            'max_usable_faces_yolov9': int(self.config.get('max_usable_faces_yolov9', 12)), # Guide
            'min_agreement_ratio': float(self.config.get('min_agreement_ratio', 0.5)), # Guide
            'min_quality_threshold': int(self.config.get('min_quality_threshold', 40)), # Guide
            'iou_threshold': float(self.config.get('iou_threshold_agreement', 0.3)) 
        }
        
        # ENHANCED VERSION: Use relaxed detection_params from config
        self.detection_params = {
            'conf_threshold': self.config.get('conf_threshold', 0.10), # Guide
            'iou_threshold': self.config.get('iou_threshold_nms', 0.35), 
            'img_size': self.config.get('img_size', 640)
        }

        # ENHANCED VERSION: Use relaxed quality_analyzer config
        quality_analyzer_config = self.config.get('quality_config', {})
        quality_analyzer_config.setdefault('min_quality_threshold', self.decision_criteria['min_quality_threshold'])
        self.quality_analyzer = FaceQualityAnalyzer(quality_analyzer_config)
        
        self.yolov9c_model_path = self.config.get('yolov9c_model_path', 'model/face-detection/yolov9c-face-lindevs.onnx')
        self.yolov9e_model_path = self.config.get('yolov9e_model_path', 'model/face-detection/yolov9e-face-lindevs.onnx')
        self.yolov11m_model_path = self.config.get('yolov11m_model_path', 'model/face-detection/yolov11m-face.pt')
        
        # ENHANCED VERSION: Add fallback_config
        self.fallback_config = self.config.get('fallback_config', get_relaxed_face_detection_config()['fallback_config'])


        self.decision_log = []
        self.models_loaded = False
    
    async def initialize(self) -> bool:
        """
        โหลดโมเดลตรวจจับใบหน้าาทั้งหมด
        
        Returns:
            สถานะการโหลดโมเดล
        """
        try:
            logger.info("กำลังโหลดโมเดลตรวจจับใบหน้าาทั้งหมด (Relaxed/Enhanced Mode)...")
            
            if self.use_enhanced_detector:
                logger.info("กำลังโหลดตัวตรวจจับใบหน้าาขั้นสูง (Enhanced Face Detector)...")
                # Assuming EnhancedDetectorAdapter is already initialized if use_enhanced_detector is true
                # Or it needs its own config path. For now, let's assume it's handled if self.use_enhanced_detector is true.
                if 'enhanced' not in self.models: # Basic check
                    self.models['enhanced'] = EnhancedDetectorAdapter(self.vram_manager) # Needs proper config
                    # init_success = await self.models['enhanced'].initialize()
                    # if not init_success: logger.warning("Enhanced detector failed to init.")

            # Load YOLO models as before, but paths come from self.config
            # YOLOv9c
            yolov9c_allocation = await self.vram_manager.request_model_allocation("yolov9c-face", "high", "face_detection_service")
            self.models['yolov9c'] = YOLOv9ONNXDetector(self.yolov9c_model_path, "YOLOv9c")
            self.models['yolov9c'].load_model("cuda" if yolov9c_allocation.location.value == "gpu" else "cpu")
            
            # YOLOv9e
            yolov9e_allocation = await self.vram_manager.request_model_allocation("yolov9e-face", "high", "face_detection_service")
            self.models['yolov9e'] = YOLOv9ONNXDetector(self.yolov9e_model_path, "YOLOv9e")
            self.models['yolov9e'].load_model("cuda" if yolov9e_allocation.location.value == "gpu" else "cpu")
            
            # YOLOv11m
            yolov11m_allocation = await self.vram_manager.request_model_allocation("yolov11m-face", "critical", "face_detection_service")
            self.models['yolov11m'] = YOLOv11Detector(self.yolov11m_model_path, "YOLOv11m")
            self.models['yolov11m'].load_model("cuda" if yolov11m_allocation.location.value == "gpu" else "cpu")
            
            self.models_loaded = True
            logger.info("โหลดโมเดลตรวจจับใบหน้าเรียบร้อยแล้ว (Relaxed/Enhanced Mode)")
            return True
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}", exc_info=True)
            return False

    # ENHANCED VERSION of detect_faces with new fallback system
    async def detect_faces(self, 
                         image_input: Union[str, np.ndarray],
                         model_name: Optional[str] = None, 
                         conf_threshold: Optional[float] = None,
                         iou_threshold: Optional[float] = None,
                         min_face_size: Optional[Tuple[int, int]] = None, # Parameter added, but not used in current logic
                         max_faces: Optional[int] = None, # Parameter added, but not used in current logic
                         return_landmarks: bool = False, # Parameter added, but not used in current logic
                         # New parameters from user prompt for this method:
                         min_quality_threshold: Optional[float] = None,
                         use_fallback: bool = True,
                         fallback_strategy: Optional[List[Dict[str, Any]]] = None,
                         force_cpu: bool = False # Parameter added, but not used in current logic
                         ) -> DetectionResult:
        """
        ตรวจจับใบหน้าในรูปภาพโดยเลือกโมเดลที่เหมาะสมที่สุดโดยอัตโนมัติ,
        พร้อมระบบ Fallback ที่ปรับปรุงใหม่ (Enhanced Detection Strategy).
        """
        start_time_total = time.time() # Define start_time_total at the beginning

        if not self.models_loaded:
            logger.warning("Models were not loaded. Attempting to initialize now...")
            initialized = await self.initialize()
            if not initialized:
                # Return DetectionResult with error, as per new structure
                return DetectionResult(faces=[], 
                                       image_shape=(0,0,0), # Provide a default shape if image is not defined
                                       total_processing_time=time.time() - start_time_total,
                                       model_used="N/A", 
                                       error="Models not loaded and initialization failed.")
            logger.info("Models initialized successfully.")
        
        # image variable will be defined after this block
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.error(f"ไม่พบไฟล์รูปภาพ: {image_input}")
                return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error=f"File not found: {image_input}")
            try:
                image = cv2.imread(image_input)
                if image is None:
                    logger.error(f"ไม่สามารถอ่านไฟล์รูปภาพ: {image_input}")
                    return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error=f"Cannot read image file: {image_input}")
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการโหลดรูปภาพ {image_input}: {e}")
                return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error=f"Error loading image: {e}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            logger.error("ประเภทข้อมูลรูปภาพไม่ถูกต้อง ต้องเป็น str หรือ np.ndarray")
            return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error="Invalid image input type")

        if image.size == 0:
             logger.error("รูปภาพที่รับเข้ามาว่างเปล่า")
             return DetectionResult(faces=[], image_shape=(image.shape if hasattr(image, 'shape') else (0,0,0)), total_processing_time=time.time()-start_time_total, model_used="N/A", error="Empty image provided")

        # Determine primary model and parameters
        # User prompt: "model_name: Optional[str] = 'auto'" - default changed here
        primary_model_name = model_name if model_name and model_name != 'auto' else self.config.get('default_model', 'yolov9c') 
        current_conf = conf_threshold if conf_threshold is not None else self.detection_params['conf_threshold']
        current_iou = iou_threshold if iou_threshold is not None else self.detection_params['iou_threshold']
        # Use provided min_quality_threshold or from service config
        current_min_quality = min_quality_threshold if min_quality_threshold is not None else self.config.get('filter_min_quality_final', 40.0)

        logger.info(f"Starting detection with primary model: {primary_model_name}, conf: {current_conf}, iou: {current_iou}, min_quality: {current_min_quality}")
        
        detected_faces_final: List[FaceDetection] = []
        model_used_for_detection = "N/A"
        detection_time_ms = 0.0
        fallback_actually_used = False
        error_message = None

        # --- Primary Detection Attempt ---
        try:
            if primary_model_name in self.models:
                detector = self.models[primary_model_name]
                model_used_for_detection = primary_model_name
                start_detect_time = time.time()

                raw_bboxes = []
                if hasattr(detector, 'detect_faces_raw') and callable(detector.detect_faces_raw):
                    # Assuming detect_faces_raw can be async or sync. If strictly async, needs await.
                    # Based on yolo_models.py, detect_faces_raw is sync.
                    if asyncio.iscoroutinefunction(detector.detect_faces_raw):
                        raw_bboxes = await detector.detect_faces_raw(image, conf_threshold=current_conf, iou_threshold=current_iou)
                    else:
                        raw_bboxes = detector.detect_faces_raw(image, conf_threshold=current_conf, iou_threshold=current_iou)
                elif hasattr(detector, 'detect') and callable(detector.detect):
                    if asyncio.iscoroutinefunction(detector.detect):
                        raw_bboxes = await detector.detect(image, conf_threshold=current_conf, iou_threshold=current_iou)
                    else:
                         raw_bboxes = detector.detect(image, conf_threshold=current_conf, iou_threshold=current_iou)
                else:
                    error_message = f"Detector {primary_model_name} has no suitable detection method (detect_faces_raw or detect)."
                    logger.error(error_message)
                
                detection_time_ms = (time.time() - start_detect_time) * 1000

                processed_faces_primary = []
                for raw_bbox_data in raw_bboxes:
                    if isinstance(raw_bbox_data, np.ndarray):
                        if len(raw_bbox_data) == 5:
                            bbox_obj = BoundingBox(x1=raw_bbox_data[0], y1=raw_bbox_data[1], x2=raw_bbox_data[2], y2=raw_bbox_data[3], confidence=raw_bbox_data[4])
                        elif len(raw_bbox_data) == 6:
                             bbox_obj = BoundingBox(x1=raw_bbox_data[0], y1=raw_bbox_data[1], x2=raw_bbox_data[2], y2=raw_bbox_data[3], confidence=raw_bbox_data[4])
                        else:
                            logger.warning(f"Skipping raw_bbox_data with unexpected length: {len(raw_bbox_data)}")
                            continue
                    elif isinstance(raw_bbox_data, BoundingBox):
                        bbox_obj = raw_bbox_data
                    else:
                        logger.warning(f"Skipping detection of unknown type: {type(raw_bbox_data)}")
                        continue
                    
                    if not validate_bounding_box(bbox_obj, image.shape[:2]):
                        logger.debug(f"Invalid bbox skipped: {bbox_obj}")
                        continue

                    quality_score = calculate_face_quality(bbox_obj, image.shape[:2])
                    
                    if quality_score >= current_min_quality:
                        face_detection_obj = FaceDetection(bbox=bbox_obj, 
                                                           quality_score=quality_score,
                                                           model_used=primary_model_name, 
                                                           processing_time=detection_time_ms / len(raw_bboxes) if raw_bboxes else detection_time_ms)
                        processed_faces_primary.append(face_detection_obj)
                
                detected_faces_final.extend(processed_faces_primary)
                logger.info(f"Primary detection ({primary_model_name}) found {len(processed_faces_primary)} valid faces (quality >= {current_min_quality}) in {detection_time_ms:.2f}ms.")

            else:
                error_message = f"Primary model {primary_model_name} not found in loaded models."
                logger.error(error_message)
        except Exception as e:
            error_message = f"Error during primary detection with {primary_model_name}: {str(e)}"
            logger.error(error_message, exc_info=True)

        # --- Fallback System (Simplified based on new structure) ---
        # Fallback if use_fallback is true AND (no faces found OR primary model failed AND error_message is set)
        if use_fallback and (not detected_faces_final or (error_message and not detected_faces_final)):
            logger.info(f"Primary detection insufficient or failed. Initiating fallback. Reason: {error_message if error_message else 'No faces found'}")
            fallback_actually_used = True
            
            # Determine fallback strategy: use provided or default from config
            active_fallback_strategy = fallback_strategy if fallback_strategy is not None else self.fallback_config.get('fallback_models', [])
            
            for fb_attempt, fb_config in enumerate(active_fallback_strategy):
                fb_model_name = fb_config.get('model_name')
                fb_conf = fb_config.get('conf_threshold', current_conf) # Use current_conf as default for fallback
                fb_iou = fb_config.get('iou_threshold', current_iou)   # Use current_iou as default for fallback
                fb_min_faces = fb_config.get('min_faces_to_accept', 1)

                logger.info(f"Fallback Attempt {fb_attempt + 1}/{len(active_fallback_strategy)}: Using {fb_model_name} (conf: {fb_conf}, iou: {fb_iou})")
                
                fb_detected_faces_current_attempt = []
                fb_detection_time_ms = 0.0
                current_fb_model_name_for_face_obj = fb_model_name # Store the name of the model used for this attempt
                
                try:
                    if fb_model_name == 'opencv_haar':
                        start_fb_detect = time.time()
                        # Get params for opencv_haar from fb_config
                        haar_scale = fb_config.get('scale_factor', 1.1)
                        haar_min_neighbors = fb_config.get('min_neighbors', 5)
                        haar_min_size = fb_config.get('min_size', (30,30))
                        raw_fb_bboxes = fallback_opencv_detection(image, scale_factor=haar_scale, min_neighbors=haar_min_neighbors, min_size=haar_min_size)
                        fb_detection_time_ms = (time.time() - start_fb_detect) * 1000
                        model_used_for_detection = "opencv_haar"
                        current_fb_model_name_for_face_obj = "opencv_haar"
                    elif fb_model_name in self.models:
                        detector_fb = self.models[fb_model_name]
                        model_used_for_detection = fb_model_name
                        current_fb_model_name_for_face_obj = fb_model_name
                        start_fb_detect = time.time()
                        
                        raw_fb_bboxes = []
                        if hasattr(detector_fb, 'detect_faces_raw') and callable(detector_fb.detect_faces_raw):
                            if asyncio.iscoroutinefunction(detector_fb.detect_faces_raw):
                                raw_fb_bboxes = await detector_fb.detect_faces_raw(image, conf_threshold=fb_conf, iou_threshold=fb_iou)
                            else:
                                raw_fb_bboxes = detector_fb.detect_faces_raw(image, conf_threshold=fb_conf, iou_threshold=fb_iou)
                        elif hasattr(detector_fb, 'detect') and callable(detector_fb.detect):
                             if asyncio.iscoroutinefunction(detector_fb.detect):
                                raw_fb_bboxes = await detector_fb.detect(image, conf_threshold=fb_conf, iou_threshold=fb_iou)
                             else:
                                raw_fb_bboxes = detector_fb.detect(image, conf_threshold=fb_conf, iou_threshold=fb_iou)
                        else:
                            logger.warning(f"Fallback detector {fb_model_name} has no suitable detection method.")
                            continue

                        fb_detection_time_ms = (time.time() - start_fb_detect) * 1000
                    else:
                        logger.warning(f"Fallback model {fb_model_name} not found.")
                        continue

                    # Process raw_fb_bboxes
                    for raw_bbox_data_fb in raw_fb_bboxes:
                        if isinstance(raw_bbox_data_fb, np.ndarray):
                            if len(raw_bbox_data_fb) == 5:
                                bbox_obj_fb = BoundingBox(x1=raw_bbox_data_fb[0], y1=raw_bbox_data_fb[1], x2=raw_bbox_data_fb[2], y2=raw_bbox_data_fb[3], confidence=raw_bbox_data_fb[4])
                            elif len(raw_bbox_data_fb) == 6:
                                bbox_obj_fb = BoundingBox(x1=raw_bbox_data_fb[0], y1=raw_bbox_data_fb[1], x2=raw_bbox_data_fb[2], y2=raw_bbox_data_fb[3], confidence=raw_bbox_data_fb[4])
                            else:
                                continue
                        elif isinstance(raw_bbox_data_fb, BoundingBox):
                            bbox_obj_fb = raw_bbox_data_fb
                        else:
                            continue
                        
                        if not validate_bounding_box(bbox_obj_fb, image.shape[:2]):
                            continue

                        quality_score_fb = calculate_face_quality(bbox_obj_fb, image.shape[:2])
                        if quality_score_fb >= current_min_quality: # Use the same min_quality as primary for consistency
                            face_detection_obj_fb = FaceDetection(bbox=bbox_obj_fb, 
                                                                  quality_score=quality_score_fb,
                                                                  model_used=current_fb_model_name_for_face_obj, 
                                                                  processing_time=fb_detection_time_ms / len(raw_fb_bboxes) if raw_fb_bboxes else fb_detection_time_ms)
                            fb_detected_faces_current_attempt.append(face_detection_obj_fb)
                    
                    logger.info(f"Fallback ({current_fb_model_name_for_face_obj}) found {len(fb_detected_faces_current_attempt)} valid faces in {fb_detection_time_ms:.2f}ms.")
                    if len(fb_detected_faces_current_attempt) >= fb_min_faces:
                        detected_faces_final = fb_detected_faces_current_attempt
                        model_used_for_detection = current_fb_model_name_for_face_obj # Update the overall model used
                        error_message = None
                        break
                
                except Exception as e_fb:
                    logger.error(f"Error during fallback detection with {fb_model_name}: {e_fb}", exc_info=True)
            
            if not detected_faces_final and not error_message:
                error_message = "All detection attempts (primary and fallback) failed to find usable faces."

        # --- Final Result Creation ---
        total_service_time = time.time() - start_time_total
        
        # If max_faces is specified, sort by quality and take top N
        if max_faces is not None and len(detected_faces_final) > max_faces:
            detected_faces_final.sort(key=lambda f: f.quality_score if f.quality_score is not None else 0, reverse=True)
            detected_faces_final = detected_faces_final[:max_faces]

        # Create DetectionResult object using the new structure
        # model_used should be the one that produced the final set of faces
        # total_processing_time is the service time, individual face processing_time is already in FaceDetection
        final_result = self._create_result(
            processed_faces=detected_faces_final, 
            image_shape_tuple=image.shape, 
            total_service_time_seconds=total_service_time, 
            model_name_overall=model_used_for_detection, 
            was_fallback_used=fallback_actually_used,
            error_str=error_message
        )
        return final_result

    # ENHANCED VERSION of _fallback_detection (now integrated into detect_faces, this can be removed or kept for specific scenarios)
    # For now, let's assume the logic is within detect_faces. If a separate _fallback_detection is needed, it can be refactored.
    # async def _fallback_detection(self, image: np.ndarray, original_results: List[FaceDetection], decision_res: DecisionResult) -> Tuple[List[FaceDetection], str, float]:
    # ... (This logic is now part of the main detect_faces method's fallback loop) ....


    # FIXED VERSION: _create_result method updated
    def _create_result(self, 
                       processed_faces: List[FaceDetection], 
                       image_shape_tuple: Tuple[int, int, int],
                       total_service_time_seconds: float, 
                       model_name_overall: str, 
                       was_fallback_used: bool, 
                       error_str: Optional[str] = None
                       ) -> DetectionResult:
        """Helper method to create DetectionResult object with new structure."""
        return DetectionResult(
            faces=processed_faces,
            image_shape=image_shape_tuple,
            total_processing_time=total_service_time_seconds * 1000, 
            model_used=model_name_overall,
            fallback_used=was_fallback_used,
            error=error_str
        )

    async def cleanup(self):
        """Clean up face detection service resources"""
        try:
            self.logger.info("🧹 Cleaning up Face Detection Service...")
            
            # Cleanup YOLO models
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'cleanup') and callable(model.cleanup):
                        if asyncio.iscoroutinefunction(model.cleanup):
                            await model.cleanup()
                        else:
                            model.cleanup()
                    self.logger.info(f"✅ Cleaned up model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error cleaning up model {model_name}: {e}")
            
            # Clear model dictionaries
            self.models.clear()
            self.model_stats.clear()
            
            # Reset state
            self.models_loaded = False
            
            # Clean up VRAM allocations
            if self.vram_manager:
                try:
                    # Release any allocated VRAM
                    await self.vram_manager.release_model_allocation("yolov9c-face", "face_detection_service")
                    await self.vram_manager.release_model_allocation("yolov9e-face", "face_detection_service")
                    await self.vram_manager.release_model_allocation("yolov11m-face", "face_detection_service")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error releasing VRAM allocations: {e}")
            
            self.logger.info("✅ Face Detection Service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Face Detection Service cleanup: {e}")