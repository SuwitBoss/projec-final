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
    คำนวณคุณภาพของใบหน้าโดยพิจารณาจากขนาด ตำแหน่ง และความมั่นใจ
    
    Args:
        detection: ข้อมูล BoundingBox ของใบหน้า
        image_shape: ขนาดรูปภาพ (height, width)
    
    Returns:
        คะแนนคุณภาพ 0-100
    """
    # น้ำหนักของแต่ละเกณฑ์
    weights = {
        'size_weight': 40,        # น้ำหนักขนาดพิกเซล (40%)
        'area_weight': 30,        # น้ำหนักพื้นที่ (30%)
        'confidence_weight': 20,  # น้ำหนัก Confidence (20%)
        'aspect_weight': 10       # น้ำหนัก Aspect Ratio (10%)
    }
    
    # เกณฑ์ขนาด
    size_thresholds = {
        'excellent': (100, 100),  # ขนาดดีเยี่ยม
        'good': (64, 64),         # ขนาดดี
        'acceptable': (32, 32),   # ขนาดพอใช้
        'minimum': (16, 16)       # ขนาดขั้นต่ำ
    }
    
    # คะแนนตามขนาด
    face_width = detection.width
    face_height = detection.height
    
    size_score = 0
    if face_width >= size_thresholds['excellent'][0] and face_height >= size_thresholds['excellent'][1]:
        size_score = 100
    elif face_width >= size_thresholds['good'][0] and face_height >= size_thresholds['good'][1]:
        size_score = 80
    elif face_width >= size_thresholds['acceptable'][0] and face_height >= size_thresholds['acceptable'][1]:
        size_score = 50
    elif face_width >= size_thresholds['minimum'][0] and face_height >= size_thresholds['minimum'][1]:
        size_score = 30
    else:
        size_score = 10
    
    # คะแนนตามสัดส่วนพื้นที่เทียบกับรูปภาพ
    image_area = image_shape[0] * image_shape[1]
    face_area = detection.area
    area_ratio = min(face_area / image_area * 100, 100)  # เปอร์เซ็นต์พื้นที่
    
    area_score = 0
    if area_ratio > 30:  # ใบหน้าใหญ่มาก
        area_score = 100
    elif area_ratio > 15:  # ใบหน้าใหญ่
        area_score = 90
    elif area_ratio > 5:  # ใบหน้าขนาดกลาง
        area_score = 75
    elif area_ratio > 1:  # ใบหน้าเล็ก
        area_score = 50
    else:  # ใบหน้าเล็กมาก
        area_score = 25
    
    # คะแนนความมั่นใจ
    confidence_score = detection.confidence * 100
    
    # คะแนนอัตราส่วน (ใบหน้าปกติมี aspect ratio ประมาณ 0.75-0.85)
    aspect_ratio = detection.aspect_ratio
    aspect_diff = abs(aspect_ratio - 0.8)
    
    aspect_score = 0
    if aspect_diff < 0.1:  # ใกล้เคียงอัตราส่วนใบหน้ามาตรฐาน
        aspect_score = 100
    elif aspect_diff < 0.2:
        aspect_score = 80
    elif aspect_diff < 0.3:
        aspect_score = 60
    elif aspect_diff < 0.5:
        aspect_score = 40
    else:
        aspect_score = 20
    
    # คำนวณคะแนนรวม
    final_score = (
        size_score * weights['size_weight'] / 100 +
        area_score * weights['area_weight'] / 100 +
        confidence_score * weights['confidence_weight'] / 100 +
        aspect_score * weights['aspect_weight'] / 100
    )
    
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
