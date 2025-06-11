# cSpell:disable
"""
ฟังก์ชันช่วยเหลือสำหรับระบบตรวจจับใบหน้า
"""
import cv2
from typing import List, Tuple, Optional, Dict, Any, Union # Ensure Union is imported
import numpy as np # Ensure numpy is imported
from dataclasses import dataclass # Ensure field is imported for default_factory if needed - REMOVED 'field'
import logging # Ensure logging is imported
import os # Add os import

logger = logging.getLogger(__name__) # Ensure logger is defined

@dataclass
class BoundingBox:
    """คลาสสำหรับเก็บข้อมูลกรอบรอบใบหน้า"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: Optional[int] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-5) # Added max to prevent division by zero

    def to_array(self) -> np.ndarray:
        """แปลงเป็น numpy array"""
        # If class_id is present and not None, include it. Otherwise, standard 5 elements.
        if self.class_id is not None:
            return np.array([self.x1, self.y1, self.x2, self.y2, self.confidence, self.class_id])
        return np.array([self.x1, self.y1, self.x2, self.y2, self.confidence])

    @classmethod
    def from_array(cls, arr: Union[np.ndarray, 'BoundingBox']) -> 'BoundingBox':
        """สร้างจาก numpy array หรือ BoundingBox object"""
        if isinstance(arr, BoundingBox):
            return arr

        if isinstance(arr, np.ndarray):
            if len(arr) == 5: # x1, y1, x2, y2, confidence
                return cls(x1=float(arr[0]), y1=float(arr[1]), x2=float(arr[2]), y2=float(arr[3]), confidence=float(arr[4]))
            elif len(arr) == 6: # x1, y1, x2, y2, confidence, class_id
                return cls(x1=float(arr[0]), y1=float(arr[1]), x2=float(arr[2]), y2=float(arr[3]), confidence=float(arr[4]), class_id=int(arr[5]))
            else:
                raise ValueError(f"Array must have 5 or 6 elements, got {len(arr)}")
        
        raise TypeError(f"Expected numpy array or BoundingBox, got {type(arr)}")


@dataclass
class FaceDetection:
    """คลาสสำหรับเก็บข้อมูลใบหน้าที่ตรวจพบ"""
    bbox: BoundingBox
    quality_score: Optional[float] = None # Renamed from quality, made optional
    model_used: str = "" # Added
    processing_time: float = 0.0 # Added
    landmarks: Optional[np.ndarray] = None # Made optional
    embedding: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON"""
        return {
            "bbox": {
                "x1": float(self.bbox.x1),
                "y1": float(self.bbox.y1),
                "x2": float(self.bbox.x2),
                "y2": float(self.bbox.y2),
                "confidence": float(self.bbox.confidence),
                "width": float(self.bbox.width),
                "height": float(self.bbox.height),
                "center_x": float(self.bbox.center[0]),
                "center_y": float(self.bbox.center[1]),
                "area": float(self.bbox.area),
                "aspect_ratio": float(self.bbox.aspect_ratio)
            },
            "quality_score": self.quality_score,
            "model_used": self.model_used,
            "processing_time": self.processing_time
        }


@dataclass
class DetectionResult:
    """คลาสสำหรับเก็บผลลัพธ์การตรวจจับใบหน้าทั้งหมด"""
    faces: List[FaceDetection] # Renamed from detections
    image_shape: Tuple[int, int, int] # Type hint changed
    total_processing_time: float # Renamed from processing_time
    model_used: str # Added
    fallback_used: bool = False # Added
    error: Optional[str] = None # Added

    @property
    def num_faces(self) -> int:
        """จำนวนใบหน้าที่พบ"""
        return len(self.faces)

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON"""
        return {
            "faces": [face.to_dict() for face in self.faces],
            "image_shape": {
                "height": self.image_shape[0],
                "width": self.image_shape[1],
                "channels": self.image_shape[2] if len(self.image_shape) > 2 else 1
            },
            "total_processing_time": self.total_processing_time,
            "face_count": len(self.faces),
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "error": self.error
        }


def calculate_face_quality(detection: BoundingBox, image_shape: Tuple[int, int]) -> float: # Signature changed
    """
    คำนวณคุณภาพของใบหน้า - FIXED VERSION
    เอา quality_weights parameter ออก
    """
    weights = {
        'size_weight': 30,
        'area_weight': 25,
        'confidence_weight': 30,
        'aspect_weight': 15
    }
    
    size_thresholds = {
        'excellent': (80, 80),
        'good': (50, 50),
        'acceptable': (25, 25),
        'minimum': (10, 10)
    }
    
    # คะแนนตามขนาด
    face_width = detection.width
    face_height = detection.height
    
    size_score = 0
    if face_width >= size_thresholds['excellent'][0] and face_height >= size_thresholds['excellent'][1]:
        size_score = 100
    elif face_width >= size_thresholds['good'][0] and face_height >= size_thresholds['good'][1]:
        size_score = 85
    elif face_width >= size_thresholds['acceptable'][0] and face_height >= size_thresholds['acceptable'][1]:
        size_score = 65
    elif face_width >= size_thresholds['minimum'][0] and face_height >= size_thresholds['minimum'][1]:
        size_score = 45
    else:
        size_score = 25
    
    # คะแนนตามสัดส่วนพื้นที่
    # Ensure image_shape has at least 2 elements for area calculation
    if len(image_shape) < 2:
        logger.warning(f"image_shape too short for area calculation: {image_shape}")
        image_area = 1 # Avoid division by zero, though this indicates an issue
    else:
        image_area = image_shape[0] * image_shape[1]

    face_area = detection.area
    area_ratio = min(face_area / max(image_area, 1e-6) * 100, 100) # max to prevent division by zero
    
    area_score = 0
    if area_ratio > 20:
        area_score = 100
    elif area_ratio > 10:
        area_score = 90
    elif area_ratio > 3:
        area_score = 80
    elif area_ratio > 0.5:
        area_score = 60
    else:
        area_score = 40
    
    # คะแนนความมั่นใจ
    confidence_score = detection.confidence * 100
    
    # คะแนนอัตราส่วน
    aspect_ratio = detection.aspect_ratio # Already handles potential division by zero in BoundingBox
    aspect_diff = abs(aspect_ratio - 0.8)
    
    aspect_score = 0
    if aspect_diff < 0.15:
        aspect_score = 100
    elif aspect_diff < 0.3:
        aspect_score = 85
    elif aspect_diff < 0.5:
        aspect_score = 70
    elif aspect_diff < 0.8:
        aspect_score = 55
    else:
        aspect_score = 35
    
    # คำนวณคะแนนรวม
    final_score = (
        size_score * weights['size_weight'] / 100 +
        area_score * weights['area_weight'] / 100 +
        confidence_score * weights['confidence_weight'] / 100 +
        aspect_score * weights['aspect_weight'] / 100
    )
    
    # เพิ่ม bonus score
    bonus_score = 5.0
    final_score = min(final_score + bonus_score, 100.0)
    
    return final_score


def draw_detection_results(image: np.ndarray, detections: List[FaceDetection], 
                         show_quality: bool = True) -> np.ndarray:
    """
    วาดกรอบรอบใบหน้าที่ตรวจพบลงบนรูปภาพ
    
    Args:
        image: รูปภาพต้นฉบับ
        detections: รายการใบหน้าที่ตรวจพบ
        show_quality: แสดงคะแนนคุณภาพหรือไม่
    
    Returns:
        รูปภาพที่วาดกรอบแล้ว
    """
    # สร้างสำเนารูปภาพ
    img_draw = image.copy()
    
    for face in detections:
        # สีตามคะแนนคุณภาพ
        if show_quality and face.quality_score is not None:
            if face.quality_score >= 80:
                color = (0, 255, 0)  # เขียว = คุณภาพดีมาก
            elif face.quality_score >= 60:
                color = (0, 255, 255)  # เหลือง = คุณภาพดี
            elif face.quality_score >= 40:
                color = (0, 165, 255)  # ส้ม = คุณภาพปานกลาง
            else:
                color = (0, 0, 255)  # แดง = คุณภาพต่ำ
        else:
            color = (0, 255, 0)  # เขียว (default)
        
        # วาดกรอบ
        x1, y1, x2, y2 = int(face.bbox.x1), int(face.bbox.y1), int(face.bbox.x2), int(face.bbox.y2)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # แสดงความมั่นใจ
        conf_text = f"{face.bbox.confidence:.2f}"
        cv2.putText(img_draw, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # แสดงคะแนนคุณภาพ
        if show_quality and face.quality_score is not None:
            quality_text = f"Q: {face.quality_score:.0f}"
            cv2.putText(img_draw, quality_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_draw


def save_detection_image(image: np.ndarray, detections: List[FaceDetection], 
                        output_dir: str, filename: str) -> str:
    """
    บันทึกรูปภาพที่มีการวาดกรอบใบหน้าแล้ว
    
    Args:
        image: รูปภาพต้นฉบับ
        detections: รายการใบหน้าที่ตรวจพบ
        output_dir: โฟลเดอร์สำหรับบันทึกไฟล์
        filename: ชื่อไฟล์
        
    Returns:
        พาธของไฟล์ที่บันทึก
    """
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(output_dir, exist_ok=True)
    
    # วาดกรอบใบหน้า
    img_with_detections = draw_detection_results(image, detections, show_quality=True)
    
    # สร้างชื่อไฟล์
    file_path = os.path.join(output_dir, filename)
    
    # บันทึกไฟล์
    cv2.imwrite(file_path, img_with_detections)
    
    return file_path


def validate_bounding_box(bbox: BoundingBox, image_shape: Tuple[int, int]) -> bool: # Signature changed
    """
    ตรวจสอบความถูกต้องของ bounding box - FIXED VERSION
    เอา allow_adjustment และ relaxed_validation parameters ออก
    """
    try:
        # Extract coordinates
        # BoundingBox object will always have x1, x1, x2, y2
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        
        # Ensure image_shape has at least 2 elements
        if len(image_shape) < 2:
            logger.warning(f"image_shape too short for validation: {image_shape}")
            return False
        img_height, img_width = image_shape[:2]
        
        # เกณฑ์การตรวจสอบ
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0: # x2, y2 can be negative if width/height is negative
            return False
        
        margin = 5 # pixels
        # Check if box is outside image boundaries (with margin)
        if x1 > img_width + margin or x2 > img_width + margin or \
           y1 > img_height + margin or y2 > img_height + margin:
             # This check might be too strict if x1,y1 can be slightly outside.
             # The original check was x2 > img_width + margin or y2 > img_height + margin
             # Let's stick to the user's provided logic for now.
             pass # User's logic is: if x2 > img_width + margin or y2 > img_height + margin: return False

        if x2 > img_width + margin or y2 > img_height + margin: # User's specific check
            return False

        if x2 <= x1 or y2 <= y1: # Invalid box dimensions
            return False
        
        width = x2 - x1
        height = y2 - y1
        if width < 8 or height < 8: # Minimum size
            return False
        
        bbox_area = width * height
        image_area = img_width * img_height
        if image_area == 0: # Avoid division by zero if image area is zero
            return False 
        area_ratio = bbox_area / image_area
        
        # Max area ratio (e.g., face shouldn't be 98% of the image)
        if area_ratio > 0.98: # User specified 0.98, previous was 0.95
            return False
        
        # Aspect ratio constraints
        if height == 0: # Avoid division by zero for aspect ratio
            return False
        aspect_ratio = width / height
        if aspect_ratio < 0.1 or aspect_ratio > 15.0: # User specified 0.1 and 15.0
            return False
        
        return True
        
    except Exception as e:
        # Use logger if available, otherwise print
        if 'logger' in globals():
            logger.error(f"Bbox validation failed: {e} for bbox {bbox} and image_shape {image_shape}")
        else:
            print(f"Bbox validation failed: {e} for bbox {bbox} and image_shape {image_shape}")
        return False


def filter_detection_results(faces: list, image_shape: Tuple[int, int], 
                           min_quality: float = 30.0) -> list:  # ลดจาก 50.0 เป็น 30.0
    """
    กรองผลลัพธ์การตรวจจับตามคุณภาพและความถูกต้อง - RELAXED VERSION
    
    Args:
        faces: รายการใบหน้าที่ตรวจพบ
        image_shape: (height, width) ของรูปภาพ
        min_quality: คะแนนคุณภาพขั้นต่ำ (ลดลง)
    
    Returns:
        รายการใบหน้าที่ผ่านการกรอง
    """
    if not faces:
        return faces
    
    filtered_faces = []
    relaxed_validation_count = 0
    
    for face in faces:
        try:            # ตรวจสอบ bounding box
            if not validate_bounding_box(face.bbox, image_shape): # Uses relaxed validate_bounding_box
                logger.debug("🚫 Face filtered: invalid bbox")
                continue
            
            # คำนวณคุณภาพใหม่ถ้าจำเป็น
            if face.quality_score is None or face.quality_score > 100: # Uses relaxed calculate_face_quality
                face.quality_score = calculate_face_quality(face.bbox, image_shape)
            
            # กรองตามคุณภาพแบบหลวม
            if face.quality_score >= min_quality:
                filtered_faces.append(face)
            else:
                # ถ้าคุณภาพต่ำแต่ confidence สูง ให้ผ่านได้
                if hasattr(face.bbox, 'confidence') and face.bbox.confidence > 0.7:
                    logger.debug(f"🎯 Low quality but high confidence face accepted: "
                               f"quality={face.quality_score:.1f}, conf={face.bbox.confidence:.3f}")
                    filtered_faces.append(face)
                    relaxed_validation_count += 1
                else:
                    logger.debug(f"🚫 Face filtered: quality {face.quality_score:.1f} < {min_quality}")
                
        except Exception as e:
            logger.error(f"❌ Error filtering face: {e}")
            # ในกรณีเกิดข้อผิดพลาด ให้ใส่ใบหน้านี้ไปด้วย (relaxed approach)
            logger.debug("🔄 Adding face despite filtering error (relaxed mode)")
            filtered_faces.append(face)
            continue
    
    if relaxed_validation_count > 0:
        logger.info(f"🎯 Relaxed validation allowed {relaxed_validation_count} additional faces")
    
    logger.info(f"🎯 Filtered faces: {len(faces)} -> {len(filtered_faces)}")
    return filtered_faces
