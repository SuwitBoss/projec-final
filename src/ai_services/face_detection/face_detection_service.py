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

from .yolo_models import YOLOv9ONNXDetector, YOLOv11Detector
from .utils import BoundingBox, FaceDetection, DetectionResult, calculate_face_quality, filter_detection_results
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
            # Default confidence for Haar, class_id can be set to a default (e.g., 0 for face)
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.5, class_id=0) 
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
            config: การตั้งค่าสำหรับบริการ. If None, uses get_relaxed_face_detection_config().
        """
        self.vram_manager = vram_manager
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
                         min_face_size: Optional[Tuple[int, int]] = None, 
                         max_faces: Optional[int] = None, 
                         return_landmarks: bool = False) -> DetectionResult:
        """
        ตรวจจับใบหน้าในรูปภาพโดยเลือกโมเดลที่เหมาะสมที่สุดโดยอัตโนมัติ,
        พร้อมระบบ Fallback ที่ปรับปรุงใหม่ (Enhanced Detection Strategy).
        """
        if not self.models_loaded:
            logger.warning("Models were not loaded. Attempting to initialize now...")
            initialized = await self.initialize()
            if not initialized:
                raise RuntimeError("โมเดลยังไม่ได้ถูกโหลด และการโหลดล้มเหลว กรุณาเรียก initialize() ก่อน")
            logger.info("Models initialized successfully.")

        start_time_total = time.time()
        
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.error(f"ไม่พบไฟล์รูปภาพ: {image_input}")
                return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error_message=f"File not found: {image_input}")
            try:
                image = cv2.imread(image_input)
                if image is None:
                    logger.error(f"ไม่สามารถอ่านไฟล์รูปภาพ: {image_input}")
                    return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error_message=f"Cannot read image file: {image_input}")
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการโหลดรูปภาพ {image_input}: {e}")
                return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error_message=f"Error loading image: {e}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            logger.error("ประเภทข้อมูลรูปภาพไม่ถูกต้อง ต้องเป็น str หรือ np.ndarray")
            return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error_message="Invalid image input type")

        if image.size == 0:
             logger.error("รูปภาพที่รับเข้ามาว่างเปล่า")
             return DetectionResult(faces=[], image_shape=(0,0,0), total_processing_time=time.time()-start_time_total, model_used="N/A", error_message="Empty image provided")

        # Determine primary model and parameters
        primary_model_name = model_name if model_name and model_name != 'auto' else 'yolov9c' # Default to yolov9c for initial attempt
        current_conf = conf_threshold if conf_threshold is not None else self.detection_params['conf_threshold']
        current_iou = iou_threshold if iou_threshold is not None else self.detection_params['iou_threshold']

        logger.info(f"Starting detection with primary model: {primary_model_name}, conf: {current_conf}, iou: {current_iou}")
        
        detected_faces: List[FaceDetection] = []
        model_used_for_primary_detection = primary_model_name
        primary_detection_time = 0.0
        decision_res = DecisionResult() # For logging fallback attempts

        # --- Primary Detection Attempt ---
        try:
            if primary_model_name in self.models:
                detector = self.models[primary_model_name]
                start_primary_detect = time.time()
                # Assuming detect_faces_raw is the method for YOLO models
                if isinstance(detector, (YOLOv9ONNXDetector, YOLOv11Detector)):
                    raw_bboxes = detector.detect(image, conf_threshold=current_conf, iou_threshold=current_iou)
                # elif isinstance(detector, EnhancedDetectorAdapter): # If you have a unified interface
                #     raw_bboxes = await detector.detect(image, conf_threshold=current_conf) # Example
                else: # Fallback to a generic call if type is unknown but in self.models
                    logger.warning(f"Detector for {primary_model_name} is of unknown type, attempting generic detect.")
                    raw_bboxes = detector.detect(image, conf_threshold=current_conf, iou_threshold=current_iou) # Placeholder

                primary_detection_time = time.time() - start_primary_detect
                
                # Process raw_bboxes into FaceDetection objects
                for raw_bbox_array in raw_bboxes:
                    bbox_obj = BoundingBox.from_array(raw_bbox_array) # Convert np.array to BoundingBox object
                    # Calculate quality using the relaxed utils.py version
                    quality_score = calculate_face_quality(bbox_obj, image.shape[:2])
                    detected_faces.append(FaceDetection(bbox=bbox_obj, quality_score=quality_score))
                logger.info(f"Primary detection ({primary_model_name}) found {len(detected_faces)} faces in {primary_detection_time:.4f}s.")

            else: # This case should ideally not happen if model_name is validated or comes from a fixed set
                logger.error(f"Primary model {primary_model_name} not found in loaded models.")
                # Proceed to fallback if enabled
        except Exception as e:
            logger.error(f"Error during primary detection with {primary_model_name}: {e}", exc_info=True)
            # Proceed to fallback if enabled

        # --- Fallback System ---
        if self.fallback_config.get('enable_fallback_system', False) and \
           (len(detected_faces) == 0 or \
            (self.fallback_config.get('always_run_all_fallbacks_if_zero_initial', True) and len(detected_faces) == 0)):
            
            logger.info("Primary detection yielded too few results. Initiating fallback system.")
            decision_res.fallback_used = True
            
            current_fallback_attempt = 0
            max_attempts = self.fallback_config.get('max_fallback_attempts', 3)
            
            # Use a copy of detected_faces for fallback iterations to avoid modifying the primary result directly yet
            fallback_candidates = list(detected_faces)

            for attempt_num, fallback_model_config in enumerate(self.fallback_config.get('fallback_models', [])):
                if current_fallback_attempt >= max_attempts:
                    logger.info("Max fallback attempts reached.")
                    break
                
                # If we already have faces from a previous fallback that met min_detections_after_fallback,
                # and we are not forced to run all fallbacks, we can stop.
                if len(fallback_candidates) >= self.fallback_config.get('min_detections_after_fallback', 1) and \
                   not (self.fallback_config.get('always_run_all_fallbacks_if_zero_initial', True) and len(detected_faces) == 0) : # Check original detected_faces for the 'always_run' condition
                    logger.info(f"Sufficient faces ({len(fallback_candidates)}) found from fallback, stopping further fallbacks.")
                    break

                current_fallback_attempt += 1
                fb_model_name = fallback_model_config['model_name']
                fb_conf = fallback_model_config.get('conf_threshold', self.detection_params['conf_threshold'])
                fb_iou = fallback_model_config.get('iou_threshold', self.detection_params['iou_threshold'])
                fb_min_faces = fallback_model_config.get('min_faces_to_accept', 1)

                attempt_info = {'model_name': fb_model_name, 'conf': fb_conf, 'iou': fb_iou, 'attempt': current_fallback_attempt}
                logger.info(f"Fallback Attempt {current_fallback_attempt}/{max_attempts} using {fb_model_name} (Conf: {fb_conf}, IoU: {fb_iou})")

                try:
                    fb_detected_this_attempt = []
                    start_fb_detect = time.time()
                    if fb_model_name == 'opencv_haar':
                        haar_scale = fallback_model_config.get('scale_factor', 1.1)
                        haar_neighbors = fallback_model_config.get('min_neighbors', 3)
                        haar_min_size = fallback_model_config.get('min_size', (20,20))
                        raw_fb_bboxes = fallback_opencv_detection(image, scale_factor=haar_scale, min_neighbors=haar_neighbors, min_size=haar_min_size)
                    elif fb_model_name in self.models:
                        detector = self.models[fb_model_name]
                        if isinstance(detector, (YOLOv9ONNXDetector, YOLOv11Detector)):
                             raw_fb_bboxes = detector.detect(image, conf_threshold=fb_conf, iou_threshold=fb_iou)
                        # Add elif for EnhancedDetectorAdapter if it has a compatible method
                        else:
                            logger.warning(f"Fallback model {fb_model_name} is of an unsupported type for direct call, skipping.")
                            raw_fb_bboxes = []
                    else:
                        logger.warning(f"Fallback model {fb_model_name} not loaded, skipping.")
                        raw_fb_bboxes = []
                    
                    fb_detect_time = time.time() - start_fb_detect
                    attempt_info['time'] = fb_detect_time

                    for raw_bbox_array in raw_fb_bboxes:
                        bbox_obj = BoundingBox.from_array(raw_bbox_array) # Convert np.array to BoundingBox object
                        quality_score = calculate_face_quality(bbox_obj, image.shape[:2])
                        fb_detected_this_attempt.append(FaceDetection(bbox=bbox_obj, quality_score=quality_score))
                    
                    logger.info(f"Fallback {fb_model_name} found {len(fb_detected_this_attempt)} faces in {fb_detect_time:.4f}s.")
                    attempt_info['faces_found'] = len(fb_detected_this_attempt)

                    # Logic to combine/replace detections. For now, if primary was empty, take these.
                    # More sophisticated merging (e.g. NMS across primary and fallback) could be added.
                    if len(fallback_candidates) < fb_min_faces and len(fb_detected_this_attempt) >= fb_min_faces:
                        logger.info(f"Fallback {fb_model_name} provided {len(fb_detected_this_attempt)} faces, replacing previous {len(fallback_candidates)} candidates.")
                        fallback_candidates = fb_detected_this_attempt # Replace if this fallback is better
                        model_used_for_primary_detection = f"{primary_model_name} -> {fb_model_name} (Fallback)" # Update model string
                        primary_detection_time += fb_detect_time # Add time
                    elif len(fb_detected_this_attempt) > len(fallback_candidates): # Simple: if more faces, prefer this set
                        logger.info(f"Fallback {fb_model_name} found more faces ({len(fb_detected_this_attempt)}) than current candidates ({len(fallback_candidates)}). Updating.")
                        fallback_candidates = fb_detected_this_attempt
                        model_used_for_primary_detection = f"{primary_model_name} -> {fb_model_name} (Fallback)"
                        primary_detection_time += fb_detect_time


                except Exception as e_fb:
                    logger.error(f"Error during fallback detection with {fb_model_name}: {e_fb}", exc_info=True)
                    attempt_info['error'] = str(e_fb)
                
                decision_res.fallback_attempts_info.append(attempt_info)
            
            # After all fallbacks, detected_faces should be the result of the best fallback attempt (or primary if no fallback was better/triggered)
            detected_faces = fallback_candidates # Update detected_faces with the final list from fallbacks

        # --- Final Processing and Result Creation ---
        # The _create_result method will handle filtering by quality, max_faces etc.
        # The min_quality for filtering is now taken from 'filter_min_quality_final' in config.
        final_result = self._create_result(
            detected_faces, 
            image.shape, 
            model_used_for_primary_detection, 
            primary_detection_time, # This time might now include fallback time
            time.time() - start_time_total, # Total processing time for the whole function
            max_faces=max_faces,
            decision_log_override=decision_res # Pass the decision result with fallback info
        )
        
        logger.info(f"Final detection result: {len(final_result.faces)} faces using {final_result.model_used}. Total time: {final_result.total_processing_time:.4f}s")
        if decision_res.fallback_used:
            logger.info(f"Fallback summary: {len(decision_res.fallback_attempts_info)} attempts made.")

        return final_result

    # ENHANCED VERSION of _fallback_detection (now integrated into detect_faces, this can be removed or kept for specific scenarios)
    # For now, let's assume the logic is within detect_faces. If a separate _fallback_detection is needed, it can be refactored.
    # async def _fallback_detection(self, image: np.ndarray, original_results: List[FaceDetection], decision_res: DecisionResult) -> Tuple[List[FaceDetection], str, float]:
    # ... (This logic is now part of the main detect_faces method's fallback loop) ....


    # ENHANCED VERSION of _create_result
    def _create_result(self, 
                       faces: List[FaceDetection], 
                       image_shape: Tuple[int, int, int], 
                       model_used: str, 
                       model_processing_time: float, 
                       total_processing_time: float,
                       max_faces: Optional[int] = None,
                       decision_log_override: Optional[DecisionResult] = None # To pass fallback info
                       ) -> DetectionResult:
        """
        สร้างผลลัพธ์การตรวจจับใบหน้า, กรองตามคุณภาพ (RELAXED), และจำกัดจำนวนใบหน้า.
        """
        # Filter by quality using the 'filter_min_quality_final' from config
        # This uses the relaxed calculate_face_quality scores already on the FaceDetection objects
        min_quality_for_filtering = self.config.get('filter_min_quality_final', 40.0) # Default to guide's 40
        
        # The filter_detection_results function from utils.py (RELAXED VERSION) will be used.
        # It needs to be passed the correct min_quality.
        # We assume faces already have quality_score calculated with relaxed_validation=True.
        
        # We can directly filter here based on the quality scores already calculated.
        # The `filter_detection_results` from `utils.py` is more complex and might re-validate or adjust.
        # For simplicity here, let's filter based on the already computed (relaxed) quality scores.
        
        # Step 1: Filter by the final quality threshold
        high_quality_faces = [
            face for face in faces if face.quality_score is not None and face.quality_score >= min_quality_for_filtering
        ]
        
        logger.debug(f"Initial faces: {len(faces)}, Filtered by quality ({min_quality_for_filtering}): {len(high_quality_faces)}")

        # Step 2: If still too many faces, sort by quality and take top N (if max_faces is set)
        if max_faces is not None and len(high_quality_faces) > max_faces:
            # Sort by quality_score descending, then by confidence if scores are equal
            high_quality_faces.sort(key=lambda f: (f.quality_score or 0, f.bbox.confidence or 0), reverse=True)
            final_faces = high_quality_faces[:max_faces]
            logger.debug(f"Applied max_faces ({max_faces}): {len(final_faces)} faces.")
        else:
            final_faces = high_quality_faces
            # Optionally sort them anyway if an order is preferred
            final_faces.sort(key=lambda f: (f.quality_score or 0, f.bbox.confidence or 0), reverse=True)


        # Create DetectionResult object
        result = DetectionResult(
            faces=final_faces,
            image_shape=image_shape,
            model_used=model_used,
            # model_processing_time=model_processing_time, # Removed: Not an expected __init__ argument
            total_processing_time=total_processing_time,
            # num_faces=len(final_faces), # Removed: Not an expected __init__ argument
            # decision_log=decision_log_override.to_dict() if decision_log_override else {}, # Removed: Not an expected __init__ argument
            fallback_used=decision_log_override.fallback_used if decision_log_override and hasattr(decision_log_override, 'fallback_used') else False
        )
        
        # Attributes like num_faces and decision_log (if they exist on DetectionResult)
        # would be set by other parts of the code or need to be handled if they are not init args.
        # The primary fix here is to correct the __init__ call.

        # Analyze quality of the *final* set of faces
        if final_faces: # Only analyze if there are faces
            quality_analysis = self.quality_analyzer.analyze_detection_quality(final_faces)
            result.quality_info = quality_analysis
            logger.info(f"Final quality analysis: Usable={quality_analysis.get('usable_count')}/{quality_analysis.get('total_count')}, AvgQ={quality_analysis.get('avg_quality'):.2f}")
        else:
            result.quality_info = self.quality_analyzer.analyze_detection_quality([]) # Get empty structure

        return result

    async def get_service_info(self) -> Dict[str, Any]:
        """
        ดูข้อมูลของบริการตรวจจับใบหน้า
        
        Returns:
            ข้อมูลบริการและสถิติการใช้งาน
        """
        vram_status = await self.vram_manager.get_vram_status()
        
        # สถิติการตัดสินใจ
        decision_stats = {}
        if self.decision_log:
            # จำนวนการตัดสินใจทั้งหมด
            decision_stats["total_decisions"] = len(self.decision_log)
            
            # จำนวนครั้งที่ใช้แต่ละโมเดล
            model_counts = {"yolov9c": 0, "yolov9e": 0, "yolov11m": 0}
            for decision in self.decision_log:
                model_used = decision["step4_results"]["model_used"]
                model_counts[model_used] = model_counts.get(model_used, 0) + 1
            
            decision_stats["model_usage"] = model_counts
            
            # สถิติความเห็นด้วย
            agreement_counts = {"high_overlap": 0, "low_overlap": 0}
            for decision in self.decision_log:
                agreement_type = decision["step2_agreement"]["type"]
                agreement_counts[agreement_type] = agreement_counts.get(agreement_type, 0) + 1
            
            decision_stats["agreement_stats"] = agreement_counts
            
            # เหตุผลการตัดสินใจใช้ YOLOv11m
            yolov11m_reasons = {}
            for decision in self.decision_log:
                if decision["step3_decision"]["use_yolov11m"]:
                    for reason in decision["step3_decision"]["reasons"]:
                        yolov11m_reasons[reason] = yolov11m_reasons.get(reason, 0) + 1
            
            decision_stats["yolov11m_reasons"] = yolov11m_reasons
            
            # สถิติคุณภาพ
            quality_stats = {
                "total_faces": 0,
                "usable_faces": 0,
                "avg_quality_ratio": 0.0
            }
            
            for decision in self.decision_log:
                if "quality_info" in decision:
                    quality_stats["total_faces"] += decision["quality_info"].get("total_count", 0)
                    quality_stats["usable_faces"] += decision["quality_info"].get("usable_count", 0)
            
            if quality_stats["total_faces"] > 0:
                quality_stats["avg_quality_ratio"] = quality_stats["usable_faces"] / quality_stats["total_faces"] * 100
            
            decision_stats["quality_stats"] = quality_stats
        
        return {
            "service_name": "Enhanced Intelligent Face Detection Service",
            "models_loaded": self.models_loaded,
            "available_models": list(self.models.keys()),
            "model_stats": self.model_stats,
            "decision_criteria": self.decision_criteria,
            "detection_params": self.detection_params,
            "vram_status": vram_status,
            "decision_stats": decision_stats,
            "recent_decisions": self.decision_log[-5:] if self.decision_log else []
        }
    
    async def cleanup(self):
        """
        ล้างทรัพยากรและปล่อย VRAM
        """
        logger.info("กำลังล้างทรัพยากร FaceDetectionService...")
        for model_name, model_instance in self.models.items():
            try:
                if hasattr(model_instance, 'cleanup'): # For YOLOv11Detector or others with cleanup
                    model_instance.cleanup()
                # For ONNX models, explicit cleanup might not be needed beyond releasing allocation
                
                # Release VRAM allocation
                # The allocation object might not be stored directly on the model instance in this structure.
                # This part needs to align with how allocations are tracked.
                # Assuming a naming convention for release:
                await self.vram_manager.release_model_allocation(f"{model_name}-face", "face_detection_service")
                logger.info(f"Released VRAM for {model_name}")
            except Exception as e:
                logger.error(f"Error during cleanup for model {model_name}: {e}")
        
        self.models.clear()
        self.models_loaded = False
        logger.info("ล้างทรัพยากร FaceDetectionService เสร็จสิ้น")

    async def enhanced_intelligent_detect(self, 
                                          image: np.ndarray, 
                                          conf_threshold: float, 
                                          iou_threshold: float,
                                          return_landmarks: bool = False) -> DetectionResult:
        """
        ใช้ EnhancedDetectorAdapter ถ้าเปิดใช้งาน หรือ fallback ไปที่ intelligent_detect.
        This method now primarily acts as a wrapper or decision point.
        The core detection, including fallbacks, is handled by detect_faces.
        """
        start_total_time = time.time()

        if self.use_enhanced_detector and 'enhanced' in self.models and isinstance(self.models['enhanced'], EnhancedDetectorAdapter):
            logger.info("Using EnhancedDetectorAdapter for detection.")
            # The EnhancedDetectorAdapter might have its own complex logic or simple detection.
            # We need to ensure its output is compatible (List[FaceDetection]) and then create a result.
            # This is a simplified call; the adapter might need more specific parameters.
            try:
                # Assuming the adapter's detect method returns raw bboxes or FaceDetection objects
                # Let's assume it returns raw bboxes for consistency with how YOLO results are handled initially
                enhanced_adapter = self.models['enhanced']
                # The adapter's detect method signature might vary. This is an example.
                # raw_bboxes = await enhanced_adapter.detect(image, confidence_threshold=conf_threshold) # Example call
                
                # For now, let's assume we call the main detect_faces with 'enhanced' model type
                # if we want to use its full pipeline including quality calculation and filtering.
                # However, 'enhanced' is not a standard model in the fallback list.
                # This suggests 'enhanced_intelligent_detect' might be a separate path.

                # Let's make this simpler: if use_enhanced_detector, we call its specific detection method.
                # The result processing should be similar to _create_result.
                
                # This part is a bit ambiguous based on current structure.
                # For now, let's assume if detect_faces handles 'enhanced' correctly, we don't need extra logic here.
                logger.warning("enhanced_intelligent_detect logic needs review for integration with the main detect_faces flow.")
                return await self.detect_faces(image, model_name='enhanced' if self.use_enhanced_detector else None, 
                                               conf_threshold=conf_threshold, iou_threshold=iou_threshold, 
                                               return_landmarks=return_landmarks)

            except Exception as e:
                logger.error(f"Error during enhanced detection: {e}. Falling back to standard intelligent detection.", exc_info=True)
                # Fallback to standard intelligent detection logic (which is now part of detect_faces)
                return await self.detect_faces(image, model_name=None, # Let detect_faces decide primary
                                               conf_threshold=conf_threshold, iou_threshold=iou_threshold, 
                                               return_landmarks=return_landmarks)
        else:
            # Standard intelligent detection (now primarily handled by detect_faces with its internal logic or _intelligent_detect)
            logger.info("Using standard detection logic (fallback system integrated).")
            # Call the main detect_faces, which will use its primary model logic and fallbacks
            return await self.detect_faces(image, model_name=None, # Let detect_faces decide primary
                                           conf_threshold=conf_threshold, iou_threshold=iou_threshold, 
                                           return_landmarks=return_landmarks)

# Ensure the file ends with a newline