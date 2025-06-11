# cSpell:disable
# mypy: ignore-errors
"""
Face Detection Models และ Configuration Classes
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List


class DetectionEngine(Enum):
    """Detection Engine Types"""
    YOLOV9C = "yolov9c"
    YOLOV9E = "yolov9e" 
    YOLOV11M = "yolov11m"
    OPENCV_HAAR = "opencv_haar"
    AUTO = "auto"


@dataclass
class DetectionConfig:
    """Configuration for Face Detection"""
    engine: DetectionEngine = DetectionEngine.AUTO
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    min_face_size: int = 30
    max_faces: int = 50
    use_fallback: bool = True
    fallback_strategy: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        if self.fallback_strategy is None:
            self.fallback_strategy = [
                {
                    "model_name": "yolov9e",
                    "conf_threshold": 0.4,
                    "iou_threshold": 0.4,
                    "min_faces_to_accept": 1
                },
                {
                    "model_name": "yolov11m", 
                    "conf_threshold": 0.3,
                    "iou_threshold": 0.4,
                    "min_faces_to_accept": 1
                },
                {
                    "model_name": "opencv_haar",
                    "scale_factor": 1.1,
                    "min_neighbors": 5,
                    "min_size": (30, 30),
                    "min_faces_to_accept": 1
                }
            ]


@dataclass 
class ModelInfo:
    """Information about detection model"""
    name: str
    type: str
    device: str
    input_size: tuple
    memory_usage: int  # MB
    loaded: bool = False


@dataclass
class DetectionMetrics:
    """Performance metrics for detection"""
    total_time: float
    preprocessing_time: float
    inference_time: float
    postprocessing_time: float
    model_used: str
    fallback_used: bool = False
