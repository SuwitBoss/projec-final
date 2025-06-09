# cSpell:disable
# mypy: ignore-errors
"""
Face Recognition Data Models
ข้อมูลโครงสร้างสำหรับระบบจดจำใบหน้า
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
from enum import Enum


class RecognitionModel(Enum):
    """โมเดลสำหรับการจดจำใบหน้า"""
    ADAFACE = "ADAFACE"
    ARCFACE = "ARCFACE"
    FACENET = "FACENET"


# Alias for backward compatibility
ModelType = RecognitionModel


class RecognitionQuality(Enum):
    """ระดับคุณภาพของใบหน้าสำหรับการจดจำ"""
    HIGH = "high"          # คุณภาพดี > 80%
    MEDIUM = "medium"      # คุณภาพปานกลาง 50-80%
    LOW = "low"            # คุณภาพต่ำ < 50%
    UNKNOWN = "unknown"    # ไม่สามารถประเมินได้


@dataclass
class FaceQuality:
    """คุณภาพของใบหน้าสำหรับการจดจำ"""
    score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    pose_quality: float = 0.0


# Type aliases
EmbeddingVector = np.ndarray
FaceGallery = Dict[str, Dict[str, Any]]


@dataclass
class FaceEmbedding:
    """ผลลัพธ์การสกัด embedding vector จากรูปภาพใบหน้า"""
    vector: Optional[np.ndarray] = None
    model_type: Optional[RecognitionModel] = None
    quality_score: float = 0.0
    extraction_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    # Legacy fields for backward compatibility
    embedding: Optional[np.ndarray] = field(init=False)
    model_used: Optional[RecognitionModel] = field(init=False)
    success: bool = True
    error: Optional[str] = None
    processing_time: Optional[float] = field(init=False)
    face_quality: RecognitionQuality = RecognitionQuality.UNKNOWN
    
    def __post_init__(self):
        # Set legacy fields for backward compatibility
        self.embedding = self.vector
        self.model_used = self.model_type
        self.processing_time = self.extraction_time


@dataclass
class FaceMatch:
    """ผลการจับคู่ใบหน้ากับคนในฐานข้อมูล"""
    person_id: str
    confidence: float
    embedding: Optional[FaceEmbedding] = None
    
    # Legacy fields for backward compatibility
    identity_id: str = field(init=False)
    similarity: float = field(init=False)
    is_match: bool = field(init=False)
    
    def __post_init__(self):
        # Set legacy fields for backward compatibility
        self.identity_id = self.person_id
        self.similarity = self.confidence
        self.is_match = self.confidence > 0.6  # Default threshold


@dataclass
class FaceComparisonResult:
    """ผลลัพธ์การเปรียบเทียบใบหน้าสองใบ"""
    similarity: float
    is_same_person: bool
    confidence: float
    processing_time: float
    model_used: Optional[RecognitionModel] = None
    error: Optional[str] = None
    
    # Legacy fields for backward compatibility
    is_match: bool = field(init=False)
    threshold_used: float = 0.6
    success: bool = True
    
    def __post_init__(self):
        # Set legacy fields for backward compatibility
        self.is_match = self.is_same_person


@dataclass
class FaceRecognitionResult:
    """ผลลัพธ์การจดจำใบหน้า"""
    matches: List[FaceMatch]
    best_match: Optional[FaceMatch] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    model_used: Optional[RecognitionModel] = None
    error: Optional[str] = None
    
    # Legacy fields for backward compatibility  
    embedding: Optional[np.ndarray] = None
    success: bool = True
    embedding_quality: RecognitionQuality = RecognitionQuality.UNKNOWN


@dataclass
class ModelPerformanceStats:
    """สถิติประสิทธิภาพของโมเดล Face Recognition"""
    total_embeddings_extracted: int = 0
    total_comparisons: int = 0
    total_extraction_time: float = 0.0
    total_comparison_time: float = 0.0
    average_extraction_time: float = 0.0
    average_comparison_time: float = 0.0
    
    def update_extraction_stats(self, time_taken: float) -> None:
        """
        อัปเดตสถิติการสกัด embedding
        
        Args:
            time_taken: เวลาที่ใช้ในการสกัด embedding (วินาที)
        """
        self.total_embeddings_extracted += 1
        self.total_extraction_time += time_taken
        self.average_extraction_time = self.total_extraction_time / self.total_embeddings_extracted
    
    def update_comparison_stats(self, time_taken: float) -> None:
        """
        อัปเดตสถิติการเปรียบเทียบใบหน้า
        
        Args:
            time_taken: เวลาที่ใช้ในการเปรียบเทียบใบหน้า (วินาที)
        """
        self.total_comparisons += 1
        self.total_comparison_time += time_taken
        self.average_comparison_time = self.total_comparison_time / self.total_comparisons
