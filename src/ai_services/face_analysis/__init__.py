"""
Face Analysis Service Package
Enhanced End-to-End Face Detection + Recognition
"""

from .face_analysis_service import FaceAnalysisService
from .models import (
    FaceAnalysisResult,
    FaceResult,
    AnalysisConfig
)

__all__ = [
    'FaceAnalysisService',
    'FaceAnalysisResult', 
    'FaceResult',
    'AnalysisConfig'
]
