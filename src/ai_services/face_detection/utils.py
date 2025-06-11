# cSpell:disable
"""
ฟังก์ชันช่วยเหลือสำหรับระบบตรวจจับใบหน้า
"""
import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """
    คลาสสำหรับเก็บข้อมูลกรอบรอบใบหน้า
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: Optional[int] = None  # MODIFIED: Added class_id
    
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
        return self.width / max(self.height, 1e-5)
    
    def to_array(self) -> np.ndarray:
        """แปลงเป็น numpy array"""
        return np.array([self.x1, self.y1, self.x2, self.y2, self.confidence])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BoundingBox':
        """สร้างจาก numpy array"""
        return cls(arr[0], arr[1], arr[2], arr[3], arr[4])


@dataclass
class FaceDetection:
    """
    คลาสสำหรับเก็บข้อมูลใบหน้าที่ตรวจพบ
    """
    bbox: BoundingBox
    quality_score: Optional[float] = None
    model_used: str = ""
    processing_time: float = 0.0
    landmarks: Optional[np.ndarray] = None  # e.g., 5 keypoints (x,y)
    embedding: Optional[np.ndarray] = None # Face embedding vector
    meta: Optional[Dict[str, Any]] = None # Additional metadata
    
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
    """
    คลาสสำหรับเก็บผลลัพธ์การตรวจจับใบหน้าทั้งหมด
    """
    faces: List[FaceDetection]
    image_shape: Tuple[int, int, int]
    total_processing_time: float
    model_used: str
    fallback_used: bool = False
    error: Optional[str] = None
    model_processing_time: Optional[float] = None # MODIFIED: Added model_processing_time
    
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


def calculate_face_quality(detection: BoundingBox, image_shape: Tuple[int, int]) -> float:
    """
    คำนวณคุณภาพของใบหน้า - RELAXED VERSION
    
    Args:
        detection: ข้อมูล BoundingBox ของใบหน้า
        image_shape: ขนาดรูปภาพ (height, width)
    
    Returns:
        คะแนนคุณภาพ 0-100 (ปรับให้ให้คะแนนสูงขึ้น)
    """
    # น้ำหนักของแต่ละเกณฑ์ (ปรับให้ไม่เข้มงวด)
    weights = {
        'size_weight': 30,        # ลดจาก 40
        'area_weight': 25,        # ลดจาก 30  
        'confidence_weight': 30,  # เพิ่มจาก 20
        'aspect_weight': 15       # เพิ่มจาก 10
    }
    
    # เกณฑ์ขนาด (หลวมขึ้น)
    size_thresholds = {
        'excellent': (80, 80),    # ลดจาก (100, 100)
        'good': (50, 50),         # ลดจาก (64, 64)
        'acceptable': (25, 25),   # ลดจาก (32, 32)
        'minimum': (10, 10)       # ลดจาก (16, 16)
    }
    
    # คะแนนตามขนาด (ให้คะแนนสูงขึ้น)
    face_width = detection.width
    face_height = detection.height
    
    size_score = 0
    if face_width >= size_thresholds['excellent'][0] and face_height >= size_thresholds['excellent'][1]:
        size_score = 100
    elif face_width >= size_thresholds['good'][0] and face_height >= size_thresholds['good'][1]:
        size_score = 85  # เพิ่มจาก 80
    elif face_width >= size_thresholds['acceptable'][0] and face_height >= size_thresholds['acceptable'][1]:
        size_score = 65  # เพิ่มจาก 50
    elif face_width >= size_thresholds['minimum'][0] and face_height >= size_thresholds['minimum'][1]:
        size_score = 45  # เพิ่มจาก 30
    else:
        size_score = 25  # เพิ่มจาก 10
    
    # คะแนนตามสัดส่วนพื้นที่ (ให้คะแนนสูงขึ้น)
    image_area = image_shape[0] * image_shape[1]
    face_area = detection.area
    area_ratio = min(face_area / image_area * 100, 100)
    
    area_score = 0
    if area_ratio > 20:      # ลดจาก 30
        area_score = 100
    elif area_ratio > 10:    # ลดจาก 15
        area_score = 90
    elif area_ratio > 3:     # ลดจาก 5
        area_score = 80      # เพิ่มจาก 75
    elif area_ratio > 0.5:   # ลดจาก 1
        area_score = 60      # เพิ่มจาก 50
    else:
        area_score = 40      # เพิ่มจาก 25
    
    # คะแนนความมั่นใจ
    confidence_score = detection.confidence * 100
    
    # คะแนนอัตราส่วน (หลวมขึ้น)
    aspect_ratio = detection.aspect_ratio
    aspect_diff = abs(aspect_ratio - 0.8)
    
    aspect_score = 0
    if aspect_diff < 0.15:   # เพิ่มจาก 0.1
        aspect_score = 100
    elif aspect_diff < 0.3:  # เพิ่มจาก 0.2
        aspect_score = 85    # เพิ่มจาก 80
    elif aspect_diff < 0.5:  # เพิ่มจาก 0.3
        aspect_score = 70    # เพิ่มจาก 60
    elif aspect_diff < 0.8:  # เพิ่มจาก 0.5
        aspect_score = 55    # เพิ่มจาก 40
    else:
        aspect_score = 35    # เพิ่มจาก 20
    
    # คำนวณคะแนนรวม
    final_score = (
        size_score * weights['size_weight'] / 100 +
        area_score * weights['area_weight'] / 100 +
        confidence_score * weights['confidence_weight'] / 100 +
        aspect_score * weights['aspect_weight'] / 100
    )
    
    # เพิ่ม bonus score เพื่อให้ผ่านเกณฑ์ง่ายขึ้น
    bonus_score = 5.0  # เพิ่ม 5 คะแนน
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


def validate_bounding_box(bbox: BoundingBox, image_shape: Tuple[int, int], min_size: int = 20, max_area_ratio: float = 0.95) -> bool: # MODIFIED: max_area_ratio to 0.95
    """
    ตรวจสอบความถูกต้องของ bounding box - RELAXED VERSION
    
    Args:
        bbox: BoundingBox object หรือ dict
        image_shape: (height, width) ของรูปภาพ
    
    Returns:
        True ถ้า bounding box ถูกต้อง False ถ้าผิดปกติ
    """
    try:
        # Extract coordinates
        if hasattr(bbox, 'x1'):
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        else:
            x1 = bbox.get('x1', 0)
            y1 = bbox.get('y1', 0)
            x2 = bbox.get('x2', 0)
            y2 = bbox.get('y2', 0)
        
        img_height, img_width = image_shape[:2]
        
        # ===== เกณฑ์การตรวจสอบที่หลวมขึ้น =====
        
        # 1. ตรวจสอบพิกัดไม่ติดลบ
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            logger.debug(f"❌ Negative coordinates: ({x1}, {y1}, {x2}, {y2})") # Changed to debug
            return False
        
        # 2. ตรวจสอบไม่เกินขอบเขตรูปภาพ (ให้อภัยเล็กน้อย)
        margin = 5  # อนุญาตให้เกินขอบ 5 pixels
        if x2 > img_width + margin or y2 > img_height + margin:
            logger.debug(f"⚠️ Bbox slightly exceeds image bounds: ({x1}, {y1}, {x2}, {y2}) vs ({img_width}, {img_height})")
            # ปรับพิกัดให้อยู่ในกรอบ
            x2 = min(x2, img_width)
            y2 = min(y2, img_height)
        
        # 3. ตรวจสอบ x2 > x1 และ y2 > y1
        if x2 <= x1 or y2 <= y1:
            logger.debug(f"❌ Invalid bbox dimensions: width={x2-x1}, height={y2-y1}") # Changed to debug
            return False
        
        # 4. ตรวจสอบขนาดขั้นต่ำ (ลดลงมาก สำหรับใบหน้าเล็กมาก)
        width = x2 - x1
        height = y2 - y1
        if width < 8 or height < 8:  # ลดจาก 12 เป็น 8
            logger.debug(f"🔍 Bbox very small: {width}x{height}")
            return False
          
        # 5. ตรวจสอบไม่ครอบคลุมทั้งภาพ (หลวมขึ้นมาก)
        bbox_area = width * height
        image_area = img_width * img_height
        area_ratio = bbox_area / image_area
        
        if area_ratio > 0.98:  # เพิ่มจาก 0.95 เป็น 0.98 (ยอมรับใบหน้าที่ใหญ่มาก)
            logger.debug(f"⚠️ Bbox covers large area: {area_ratio:.1%} - but allowing it")
            # ไม่ return False ทันที แต่ให้ผ่านไป
        
        # 6. ตรวจสอบ aspect ratio สมเหตุสมผล (หลวมมาก)
        aspect_ratio = width / height
        if aspect_ratio < 0.1 or aspect_ratio > 15.0:  # หลวมมาก จาก 0.1-10.0 เป็น 0.1-15.0
            logger.debug(f"🔍 Unusual aspect ratio: {aspect_ratio:.2f} - but allowing it")
            # ไม่ return False แต่ให้ผ่านไป
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bbox validation failed: {e}")
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
