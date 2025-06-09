# cSpell:disable
# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum

"""
Face Analysis Data Models
โครงสร้างข้อมูลสำหรับระบบวิเคราะห์ใบหน้าแบบครบวงจร
"""


class AnalysisMode(Enum):
    """โหมดการวิเคราะห์"""
    DETECTION_ONLY = "detection_only"          # ตรวจจับใบหน้าเท่านั้น
    RECOGNITION_ONLY = "recognition_only"      # จดจำใบหน้าเท่านั้น (ต้องมี face crops)
    FULL_ANALYSIS = "full_analysis"            # ตรวจจับ + จดจำ
    VERIFICATION = "verification"              # เปรียบเทียบใบหน้า 2 ใบ


@dataclass
class AnalysisConfig:
    """การตั้งค่าการวิเคราะห์"""
    mode: AnalysisMode = AnalysisMode.FULL_ANALYSIS
    
    # Detection settings
    detection_model: Optional[str] = None       # auto-select if None
    min_face_size: int = 32
    confidence_threshold: float = 0.5
    max_faces: int = 50
    
    # Recognition settings  
    recognition_model: Optional[str] = None     # auto-select if None
    enable_embedding_extraction: bool = True
    enable_gallery_matching: bool = True
    gallery_top_k: int = 5
    
    # Performance settings
    batch_size: int = 8
    use_quality_based_selection: bool = True
    parallel_processing: bool = True
    
    # Output settings
    return_face_crops: bool = False
    return_embeddings: bool = False
    return_detailed_stats: bool = True


@dataclass
class FaceResult:
    """ผลลัพธ์การวิเคราะห์ใบหน้า 1 ใบ (Detection + Recognition)"""
    # Detection results    bbox: Any  # BoundingBox from face_detection.models
    confidence: float
    quality_score: float
    
    # Recognition results (optional)
    embedding: Optional[Any] = None  # FaceEmbedding from face_recognition.models
    matches: Optional[List[Any]] = None  # List[FaceMatch] from face_recognition.models
    best_match: Optional[Any] = None  # FaceMatch from face_recognition.models
    
    # Additional data
    face_crop: Optional[np.ndarray] = None
    face_id: Optional[str] = None
    
    @property
    def has_identity(self) -> bool:
        """ตรวจสอบว่าจดจำตัวตนได้หรือไม่"""
        return self.best_match is not None and getattr(self.best_match, 'is_match', False)
    
    @property
    def identity(self) -> Optional[str]:
        """ดึงตัวตนที่จดจำได้"""
        if self.has_identity and self.best_match:
            return getattr(self.best_match, 'identity_id', None)
        return None
    
    @property
    def identity_name(self) -> Optional[str]:
        """ดึงชื่อตัวตนที่จดจำได้"""
        if self.has_identity and self.best_match:
            return getattr(self.best_match, 'identity_name', None)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        result = {
            'bbox': self.bbox.to_dict(),
            'confidence': float(self.confidence),
            'quality_score': float(self.quality_score),
            'has_identity': self.has_identity,
            'identity': self.identity,
            'identity_name': self.identity_name,
            'face_id': self.face_id
        }
        
        if self.embedding:
            result['embedding'] = self.embedding.to_dict()
        
        if self.matches:
            result['matches'] = [match.to_dict() for match in self.matches]
        
        if self.best_match:
            result['best_match'] = self.best_match.to_dict()
        
        if self.face_crop is not None:
            result['face_crop_shape'] = self.face_crop.shape
        
        return result


@dataclass
class FaceAnalysisResult:
    """ผลลัพธ์การวิเคราะห์ใบหน้าทั้งหมดในรูป"""
    # Input info
    image_shape: Tuple[int, int, int]
    config: AnalysisConfig
    
    # Results
    faces: List[FaceResult]
    
    # Performance metrics
    detection_time: float
    recognition_time: float
    total_time: float
    
    # Models used
    detection_model_used: Optional[str] = None
    recognition_model_used: Optional[str] = None
    
    # Statistics
    total_faces: int = 0
    usable_faces: int = 0
    identified_faces: int = 0
    
    def __post_init__(self):
        """คำนวณ statistics หลังจากสร้าง object"""
        self.total_faces = len(self.faces)
        self.usable_faces = len([f for f in self.faces if f.quality_score >= 60])
        self.identified_faces = len([f for f in self.faces if f.has_identity])
    
    @property
    def detection_success_rate(self) -> float:
        """อัตราความสำเร็จของการตรวจจับ"""
        return (self.usable_faces / self.total_faces) if self.total_faces > 0 else 0.0
    
    @property
    def recognition_success_rate(self) -> float:
        """อัตราความสำเร็จของการจดจำ"""
        return (self.identified_faces / self.usable_faces) if self.usable_faces > 0 else 0.0
    
    @property 
    def average_confidence(self) -> float:
        """ความเชื่อมั่นเฉลี่ย"""
        if not self.faces:
            return 0.0
        return sum(face.confidence for face in self.faces) / len(self.faces)
    
    @property
    def average_quality(self) -> float:
        """คุณภาพเฉลี่ย"""
        if not self.faces:
            return 0.0
        return sum(face.quality_score for face in self.faces) / len(self.faces)
    
    def get_identified_faces(self) -> List[FaceResult]:
        """ดึงเฉพาะใบหน้าที่จดจำตัวตนได้"""
        return [face for face in self.faces if face.has_identity]
    
    def get_face_by_identity(self, identity_id: str) -> Optional[FaceResult]:
        """ค้นหาใบหน้าตาม identity"""
        for face in self.faces:
            if face.identity == identity_id:
                return face
        return None
    
    def get_faces_by_quality(self, min_quality: float = 60.0) -> List[FaceResult]:
        """ดึงใบหน้าที่มีคุณภาพเหนือเกณฑ์"""
        return [face for face in self.faces if face.quality_score >= min_quality]
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        return {
            'image_shape': self.image_shape,
            'config': {
                'mode': self.config.mode.value,
                'detection_model': self.config.detection_model,
                'recognition_model': self.config.recognition_model,
                'min_face_size': self.config.min_face_size,
                'confidence_threshold': self.config.confidence_threshold,
                'max_faces': self.config.max_faces,
                'gallery_top_k': self.config.gallery_top_k
            },
            'faces': [face.to_dict() for face in self.faces],
            'performance': {
                'detection_time': float(self.detection_time),
                'recognition_time': float(self.recognition_time), 
                'total_time': float(self.total_time),
                'detection_model_used': self.detection_model_used,
                'recognition_model_used': self.recognition_model_used
            },
            'statistics': {
                'total_faces': self.total_faces,
                'usable_faces': self.usable_faces,
                'identified_faces': self.identified_faces,
                'detection_success_rate': self.detection_success_rate,
                'recognition_success_rate': self.recognition_success_rate,
                'average_confidence': self.average_confidence,
                'average_quality': self.average_quality
            }
        }


@dataclass
class BatchAnalysisResult:
    """ผลลัพธ์การวิเคราะห์หลายรูปพร้อมกัน"""
    results: List[FaceAnalysisResult]
    total_images: int
    total_faces: int
    total_identities: int
    processing_time: float
    
    @property
    def average_faces_per_image(self) -> float:
        """จำนวนใบหน้าเฉลี่ยต่อรูป"""
        return self.total_faces / self.total_images if self.total_images > 0 else 0.0
    
    @property
    def overall_success_rate(self) -> float:
        """อัตราความสำเร็จโดยรวม"""
        if not self.results:
            return 0.0
        
        total_usable = sum(r.usable_faces for r in self.results)
        total_identified = sum(r.identified_faces for r in self.results)
        
        return total_identified / total_usable if total_usable > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        return {
            'results': [result.to_dict() for result in self.results],
            'summary': {
                'total_images': self.total_images,
                'total_faces': self.total_faces,
                'total_identities': self.total_identities,
                'processing_time': float(self.processing_time),
                'average_faces_per_image': self.average_faces_per_image,
                'overall_success_rate': self.overall_success_rate
            }
        }
