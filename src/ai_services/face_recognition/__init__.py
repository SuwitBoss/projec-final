"""
Face Recognition Service Package
Enhanced Intelligent Face Recognition System with Multiple Models
"""

from .face_recognition_service import FaceRecognitionService, RecognitionConfig
from .models import (
    FaceEmbedding,
    FaceMatch,
    FaceRecognitionResult,
    FaceComparisonResult,
    ModelType,
    FaceQuality,
    RecognitionModel,
    RecognitionQuality
)
from .model_selector import FaceRecognitionModelSelector

__all__ = [
    'FaceRecognitionService',
    'RecognitionConfig',
    'FaceEmbedding',
    'FaceMatch', 
    'FaceRecognitionResult',
    'FaceComparisonResult',
    'ModelType',
    'FaceQuality',
    'RecognitionModel',
    'RecognitionQuality',
    'FaceRecognitionModelSelector'
]
