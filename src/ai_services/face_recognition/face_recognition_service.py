"""
Face Recognition Service - Corrected version with proper VRAMManager integration
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
import asyncio
from dataclasses import dataclass
import time

from .models import (
    FaceEmbedding,
    FaceMatch,
    FaceRecognitionResult,
    FaceComparisonResult,
    ModelType
)
from .model_selector import FaceRecognitionModelSelector
from ..common.vram_manager import VRAMManager


@dataclass
class RecognitionConfig:
    """Configuration for face recognition"""
    similarity_threshold: float = 0.6
    max_faces: int = 10
    quality_threshold: float = 0.3
    auto_model_selection: bool = True
    preferred_model: Optional[ModelType] = None
    enable_quality_assessment: bool = True


class FaceRecognitionService:
    """Face Recognition Service with proper VRAM management"""
    
    def __init__(
        self,
        config: Optional[RecognitionConfig] = None,
        vram_manager: Optional[VRAMManager] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or RecognitionConfig()
        
        # Initialize VRAM manager with proper configuration if not provided
        if vram_manager is None:
            vram_config = {
                "model_vram_estimates": {
                    "ADAFACE": 249 * 1024 * 1024,  # 249MB
                    "ARCFACE": 249 * 1024 * 1024,  # 249MB  
                    "FACENET": 89 * 1024 * 1024,   # 89MB
                    "default": 512 * 1024 * 1024   # 512MB default
                },
                "total_vram_mb": 6144,  # 6GB
                "zones": {
                    "critical": {"size_mb": 2048},
                    "high_priority": {"size_mb": 2560},
                    "flexible": {"size_mb": 1536}
                }
            }
            self.vram_manager = VRAMManager(vram_config)
        else:
            self.vram_manager = vram_manager
        
        # Initialize model selector with model information
        models_info = {
            "ADAFACE": {
                "size_bytes": 249 * 1024 * 1024,  # 249MB
                "accuracy": 0.85,
                "speed": 0.7
            },
            "ARCFACE": {
                "size_bytes": 249 * 1024 * 1024,  # 249MB
                "accuracy": 0.88,
                "speed": 0.6
            },
            "FACENET": {
                "size_bytes": 89 * 1024 * 1024,   # 89MB
                "accuracy": 0.80,
                "speed": 0.9            }
        }
        self.model_selector = FaceRecognitionModelSelector(self.vram_manager, models_info)
        
        # Current loaded model
        self.current_model = None
        self.current_model_type = None
        
        # Face database - เปลี่ยนให้รองรับ multiple embeddings ต่อคน
        self.face_database: Dict[str, List[FaceEmbedding]] = {}
        
        self.logger.info("Face Recognition Service initialized")
    
    async def initialize(self) -> bool:
        """Initialize the face recognition service"""
        try:
            # Select and load initial model
            if self.config.auto_model_selection:
                model_type = await self.model_selector.select_model()
            else:
                model_type = self.config.preferred_model or ModelType.FACENET
                
            success = await self.load_model(model_type)
            
            if success:
                self.logger.info(f"Service initialized with {model_type.value} model")
                return True
            else:
                self.logger.error("Failed to initialize service")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing service: {e}")
            return False
    
    async def load_model(self, model_type: ModelType) -> bool:
        """Load specific face recognition model"""
        try:
            # Check if model is already loaded
            if self.current_model_type == model_type:
                return True
            
            # Simulate model loading
            self.logger.info(f"Loading {model_type.value} model...")
            await asyncio.sleep(0.1)  # Simulate loading time
            
            self.current_model = f"{model_type.value.lower()}_model"
            self.current_model_type = model_type
            
            self.logger.info(f"Successfully loaded {model_type.value} model")
            return True
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_type.value}: {e}")
            return False
    
    async def extract_embedding(self, face_image) -> Optional[FaceEmbedding]:
        """Extract face embedding from face image"""
        try:
            start_time = time.time()
            
            # Ensure model is loaded
            if self.current_model is None:
                if not await self.initialize():
                    return None
            
            # Simulate embedding extraction
            await asyncio.sleep(0.05)
            
            # Generate fake embedding
            if self.current_model_type == ModelType.ADAFACE:
                embedding_size = 512
            elif self.current_model_type == ModelType.ARCFACE:
                embedding_size = 512
            else:  # FaceNet
                embedding_size = 128
            
            embedding_vector = np.random.randn(embedding_size).astype(np.float32)
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            
            processing_time = time.time() - start_time
            
            embedding = FaceEmbedding(
                vector=embedding_vector,
                model_type=self.current_model_type,
                quality_score=0.8,
                extraction_time=processing_time
            )
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding: {e}")
            return None
    
    async def compare_faces(self, face1, face2) -> FaceComparisonResult:
        """Compare two face images"""
        try:
            start_time = time.time()
            
            # Extract embeddings
            embedding1 = await self.extract_embedding(face1)
            embedding2 = await self.extract_embedding(face2)
            
            if embedding1 is None or embedding2 is None:
                return FaceComparisonResult(
                    similarity=0.0,
                    is_same_person=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_used=self.current_model_type
                )
            
            # Calculate similarity (cosine similarity)
            similarity = float(np.dot(embedding1.vector, embedding2.vector))
            similarity = (similarity + 1.0) / 2.0  # Convert to 0-1 range
            
            is_same_person = similarity >= self.config.similarity_threshold
            
            result = FaceComparisonResult(
                similarity=similarity,
                is_same_person=is_same_person,
                confidence=similarity,
                processing_time=time.time() - start_time,
                model_used=self.current_model_type
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error comparing faces: {e}")
            return FaceComparisonResult(
                similarity=0.0,
                is_same_person=False,
                confidence=0.0,
                processing_time=0.0,                model_used=self.current_model_type,
                error=str(e)
            )
    
    async def add_face_to_database(self, person_id: str, face_image, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add face to recognition database - รองรับ multiple embeddings ต่อคน"""
        try:
            embedding = await self.extract_embedding(face_image)
            
            if embedding is None:
                self.logger.error(f"Failed to extract embedding for person {person_id}")
                return False
            
            # Add metadata
            if metadata:
                embedding.metadata = metadata
            
            # เก็บ multiple embeddings ต่อคน แทนการเขียนทับ
            if person_id not in self.face_database:
                self.face_database[person_id] = []
            
            # เพิ่ม embedding ใหม่เข้าไปใน list
            self.face_database[person_id].append(embedding)
            
            total_embeddings = len(self.face_database[person_id])
            self.logger.info(f"Added face for person {person_id} to database (total embeddings: {total_embeddings})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding face to database: {e}")
            return False
    
    async def recognize_face(self, face_image) -> FaceRecognitionResult:
        """Recognize face against database"""
        try:
            # Extract embedding
            embedding = await self.extract_embedding(face_image)
            
            if embedding is None:
                return FaceRecognitionResult(
                    matches=[],
                    best_match=None,
                    confidence=0.0,
                    processing_time=0.0,
                    model_used=self.current_model_type
                )
              # Search in database - เปรียบเทียบกับ multiple embeddings ต่อคน
            matches = []
            for person_id, stored_embeddings in self.face_database.items():
                # หาค่า similarity สูงสุดจาก multiple embeddings ของคนนี้
                max_similarity = 0.0
                best_stored_embedding = None
                
                for stored_embedding in stored_embeddings:
                    if stored_embedding.model_type != embedding.model_type:
                        continue
                    
                    similarity = float(np.dot(embedding.vector, stored_embedding.vector))
                    similarity = (similarity + 1.0) / 2.0
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_stored_embedding = stored_embedding
                
                # ถ้าค่า similarity สูงสุดเกิน threshold ให้เพิ่มเป็น match
                if max_similarity >= self.config.similarity_threshold and best_stored_embedding:
                    match = FaceMatch(
                        person_id=person_id,
                        confidence=max_similarity,
                        embedding=best_stored_embedding
                    )
                    matches.append(match)
            
            # Sort by confidence
            matches.sort(key=lambda x: x.confidence, reverse=True)
              # Find best match
            best_match = matches[0] if matches else None
            
            result = FaceRecognitionResult(
                matches=matches,
                best_match=best_match,
                confidence=best_match.confidence if best_match else 0.0,
                processing_time=embedding.extraction_time,
                model_used=self.current_model_type
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error recognizing face: {e}")
            return FaceRecognitionResult(
                matches=[],
                best_match=None,
                confidence=0.0,
                processing_time=0.0,
                model_used=self.current_model_type,
                error=str(e)
            )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            vram_stats = await self.vram_manager.get_memory_stats()
            vram_usage = vram_stats.get('usage_percent', 0)
        except Exception:
            vram_usage = 0
            
        return {
            'current_model': self.current_model_type.value if self.current_model_type else None,
            'database_size': len(self.face_database),
            'vram_usage': vram_usage
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.face_database.clear()
            self.current_model = None
            self.current_model_type = None
            self.logger.info("Face Recognition Service cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
