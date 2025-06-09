"""
Face Analysis API - Integrated Face Detection + Recognition Endpoints
Comprehensive face analysis combining detection and recognition services
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any, Union
import numpy as np
import cv2
import base64
import io
import asyncio
from PIL import Image
import json
import time
from datetime import datetime

from ..ai_services.face_analysis.models import (
    FaceAnalysisResult, FaceResult, AnalysisConfig,
    AnalysisMode, QualityLevel
)
from ..ai_services.face_analysis.face_analysis_service import FaceAnalysisService
from ..ai_services.face_detection.models import DetectionConfig, DetectionEngine
from ..ai_services.face_recognition.models import RecognitionModel, FaceGallery
from ..core.vram_manager import VRAMManager
from ..utils.image_utils import process_image_input, validate_image_format

router = APIRouter(prefix="/api/v1/face-analysis", tags=["face-analysis"])

# Global service references (will be injected from main.py)
face_analysis_service: Optional[FaceAnalysisService] = None
vram_manager: Optional[VRAMManager] = None

def get_face_analysis_service() -> FaceAnalysisService:
    """Dependency to get face analysis service"""
    if face_analysis_service is None:
        raise HTTPException(status_code=503, detail="Face Analysis service not initialized")
    return face_analysis_service

def get_vram_manager() -> VRAMManager:
    """Dependency to get VRAM manager"""
    if vram_manager is None:
        raise HTTPException(status_code=503, detail="VRAM manager not initialized")
    return vram_manager

@router.post("/analyze")
async def analyze_faces(
    image: UploadFile = File(...),
    analyses: Optional[str] = Form(default="detection,recognition"),
    detection_engine: Optional[str] = Form(default="yolov10n"),
    recognition_model: Optional[str] = Form(default="adaface"),
    confidence_threshold: Optional[float] = Form(default=0.5),
    max_faces: Optional[int] = Form(default=10),
    return_embeddings: Optional[bool] = Form(default=False),
    database_id: Optional[str] = Form(default="default"),
    analysis_service: FaceAnalysisService = Depends(get_face_analysis_service)
):
    """
    Complete face analysis including detection and recognition
    
    Args:
        image: Input image file
        analyses: Comma-separated analysis types (detection,recognition,verification)
        detection_engine: Detection engine to use (yolov10n, yolov5s, mediapipe, insightface)
        recognition_model: Recognition model (adaface, arcface, facenet)
        confidence_threshold: Minimum confidence for detection
        max_faces: Maximum number of faces to detect
        return_embeddings: Whether to return face embeddings
        database_id: Face database ID for recognition
    """
    try:
        # Process image
        image_data = await image.read()
        image_array = process_image_input(image_data)
        
        # Parse analysis types
        analysis_types = [a.strip().lower() for a in analyses.split(",")]
        
        # Create analysis config
        config = AnalysisConfig(
            mode=AnalysisMode.COMPREHENSIVE if len(analysis_types) > 1 else AnalysisMode.DETECTION_ONLY,
            detection_config=DetectionConfig(
                engine=DetectionEngine(detection_engine.upper()),
                confidence_threshold=confidence_threshold,
                max_faces=max_faces,
                return_landmarks=True
            ),
            recognition_config={
                "model": RecognitionModel(recognition_model.upper()),
                "threshold": 0.6,
                "return_embeddings": return_embeddings
            },
            quality_level=QualityLevel.BALANCED
        )
        
        # Create face gallery if recognition is requested
        gallery = None
        if "recognition" in analysis_types:
            # TODO: Load from actual database
            gallery = {}
        
        # Perform analysis
        result = await analysis_service.analyze_faces(image_array, config, gallery)
        
        # Format response
        response_data = {
            "success": True,
            "data": {
                "faces_detected": len(result.faces),
                "faces": [],
                "image_info": {
                    "width": image_array.shape[1],
                    "height": image_array.shape[0],
                    "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1
                }
            },
            "meta": {
                "processing_time": result.processing_time,
                "timestamp": datetime.utcnow().isoformat(),
                "analyses_performed": analysis_types,
                "models_used": {
                    "detection": config.detection_config.engine.value.lower(),
                    "recognition": config.recognition_config.get("model", "").value.lower() if config.recognition_config.get("model") else None
                }
            }
        }
        
        # Process each detected face
        for i, face in enumerate(result.faces):
            face_data = {
                "face_id": f"face_{i+1:03d}",
                "bbox": {
                    "x": int(face.detection.bbox.x1),
                    "y": int(face.detection.bbox.y1),
                    "width": int(face.detection.bbox.width),
                    "height": int(face.detection.bbox.height)
                },
                "confidence": float(face.detection.confidence),
                "quality_score": float(face.quality.overall_quality)
            }
            
            # Add detection details
            if "detection" in analysis_types:
                face_data["detection"] = {
                    "confidence": float(face.detection.confidence),
                    "engine_used": face.detection.engine_used,
                    "quality_score": float(face.quality.overall_quality)
                }
                
                if face.detection.landmarks:
                    face_data["detection"]["landmarks"] = {
                        "points": [[float(p[0]), float(p[1])] for p in face.detection.landmarks.points],
                        "type": face.detection.landmarks.landmark_type
                    }
            
            # Add recognition details
            if "recognition" in analysis_types and face.recognition:
                face_data["recognition"] = {
                    "unknown": len(face.recognition.matches) == 0,
                    "confidence": float(face.recognition.embedding_quality) if face.recognition.embedding_quality else 0.0
                }
                
                if face.recognition.matches:
                    face_data["recognition"]["matches"] = [
                        {
                            "person_id": match.identity_id,
                            "similarity": float(match.similarity),
                            "confidence": float(match.confidence),
                            "is_match": match.is_match
                        }
                        for match in face.recognition.matches
                    ]
                
                if return_embeddings and face.recognition.embedding:
                    face_data["recognition"]["embedding"] = face.recognition.embedding.tolist()
            
            response_data["data"]["faces"].append(face_data)
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face analysis failed: {str(e)}")

@router.post("/analyze-base64")
async def analyze_faces_base64(
    request: Dict[str, Any],
    analysis_service: FaceAnalysisService = Depends(get_face_analysis_service)
):
    """
    Face analysis with base64 encoded image input
    
    Request format:
    {
        "image": "base64_encoded_image_data",
        "analyses": ["detection", "recognition"],
        "options": {
            "detection_engine": "yolov10n",
            "recognition_model": "adaface", 
            "confidence_threshold": 0.5,
            "return_embeddings": false
        }
    }
    """
    try:
        # Extract parameters
        image_b64 = request.get("image")
        if not image_b64:
            raise HTTPException(status_code=422, detail="Image data required")
        
        analyses = request.get("analyses", ["detection"])
        options = request.get("options", {})
        
        # Decode image
        try:
            image_data = base64.b64decode(image_b64)
            image_array = process_image_input(image_data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid image data: {str(e)}")
        
        # Create config
        config = AnalysisConfig(
            mode=AnalysisMode.COMPREHENSIVE if len(analyses) > 1 else AnalysisMode.DETECTION_ONLY,
            detection_config=DetectionConfig(
                engine=DetectionEngine(options.get("detection_engine", "YOLOV10N").upper()),
                confidence_threshold=options.get("confidence_threshold", 0.5),
                max_faces=options.get("max_faces", 10),
                return_landmarks=True
            ),
            recognition_config={
                "model": RecognitionModel(options.get("recognition_model", "ADAFACE").upper()),
                "threshold": options.get("recognition_threshold", 0.6),
                "return_embeddings": options.get("return_embeddings", False)
            } if "recognition" in analyses else None,
            quality_level=QualityLevel.BALANCED
        )
        
        # Create face gallery
        gallery = {}
        if "recognition" in analyses:
            # Load from database or use provided gallery
            known_faces = request.get("known_faces", {})
            for person_id, embedding_data in known_faces.items():
                if isinstance(embedding_data, list):
                    gallery[person_id] = {
                        'name': person_id,
                        'embeddings': [np.array(embedding_data)]
                    }
        
        # Perform analysis
        result = await analysis_service.analyze_faces(image_array, config, gallery)
        
        # Format response similar to analyze endpoint
        response_data = {
            "success": True,
            "data": {
                "faces_detected": len(result.faces),
                "faces": []
            },
            "meta": {
                "processing_time": result.processing_time,
                "timestamp": datetime.utcnow().isoformat(),
                "analyses_performed": analyses
            }
        }
        
        # Process faces
        for i, face in enumerate(result.faces):
            face_data = {
                "face_id": f"face_{i+1:03d}",
                "bbox": {
                    "x": int(face.detection.bbox.x1),
                    "y": int(face.detection.bbox.y1), 
                    "width": int(face.detection.bbox.width),
                    "height": int(face.detection.bbox.height)
                },
                "confidence": float(face.detection.confidence)
            }
            
            if "recognition" in analyses and face.recognition:
                face_data["recognition"] = {
                    "unknown": len(face.recognition.matches) == 0,
                    "matches": [
                        {
                            "person_id": match.identity_id,
                            "similarity": float(match.similarity),
                            "is_match": match.is_match
                        }
                        for match in face.recognition.matches
                    ]
                }
                
                if options.get("return_embeddings") and face.recognition.embedding:
                    face_data["recognition"]["embedding"] = face.recognition.embedding.tolist()
            
            response_data["data"]["faces"].append(face_data)
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/verify")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model: Optional[str] = Form(default="adaface"),
    threshold: Optional[float] = Form(default=0.6),
    analysis_service: FaceAnalysisService = Depends(get_face_analysis_service)
):
    """
    1:1 face verification between two images
    
    Args:
        image1: First image file
        image2: Second image file  
        model: Recognition model to use
        threshold: Similarity threshold for match
    """
    try:
        # Process both images
        image1_data = await image1.read()
        image2_data = await image2.read()
        
        image1_array = process_image_input(image1_data)
        image2_array = process_image_input(image2_data)
        
        # Analysis config for embedding extraction
        config = AnalysisConfig(
            mode=AnalysisMode.RECOGNITION_ONLY,
            detection_config=DetectionConfig(
                engine=DetectionEngine.YOLOV10N,
                confidence_threshold=0.5,
                max_faces=1
            ),
            recognition_config={
                "model": RecognitionModel(model.upper()),
                "threshold": threshold,
                "return_embeddings": True
            },
            quality_level=QualityLevel.HIGH
        )
        
        # Analyze both images
        result1 = await analysis_service.analyze_faces(image1_array, config)
        result2 = await analysis_service.analyze_faces(image2_array, config)
        
        if not result1.faces or not result2.faces:
            return JSONResponse(content={
                "success": False,
                "error": "Face not detected in one or both images",
                "data": {
                    "faces_detected_image1": len(result1.faces),
                    "faces_detected_image2": len(result2.faces)
                }
            })
        
        # Get embeddings
        face1 = result1.faces[0]
        face2 = result2.faces[0]
        
        if not (face1.recognition and face1.recognition.embedding is not None and
                face2.recognition and face2.recognition.embedding is not None):
            return JSONResponse(content={
                "success": False,
                "error": "Failed to extract embeddings from one or both faces"
            })
        
        # Calculate similarity
        embedding1 = face1.recognition.embedding
        embedding2 = face2.recognition.embedding
        
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        
        is_same_person = similarity >= threshold
        
        response_data = {
            "success": True,
            "data": {
                "is_same_person": is_same_person,
                "similarity_score": float(similarity),
                "confidence": float(similarity),
                "threshold_used": threshold,
                "face1": {
                    "detected": True,
                    "quality_score": float(face1.quality.overall_quality),
                    "bbox": {
                        "x": int(face1.detection.bbox.x1),
                        "y": int(face1.detection.bbox.y1),
                        "width": int(face1.detection.bbox.width),
                        "height": int(face1.detection.bbox.height)
                    }
                },
                "face2": {
                    "detected": True,
                    "quality_score": float(face2.quality.overall_quality),
                    "bbox": {
                        "x": int(face2.detection.bbox.x1),
                        "y": int(face2.detection.bbox.y1),
                        "width": int(face2.detection.bbox.width),
                        "height": int(face2.detection.bbox.height)
                    }
                }
            },
            "meta": {
                "model_used": model,
                "processing_time": result1.processing_time + result2.processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze_faces(
    request: Dict[str, Any],
    analysis_service: FaceAnalysisService = Depends(get_face_analysis_service)
):
    """
    Batch analysis of multiple images
    
    Request format:
    {
        "images": [
            {"id": "img_001", "data": "base64_encoded_image1"},
            {"id": "img_002", "data": "base64_encoded_image2"}
        ],
        "analyses": ["detection", "recognition"],
        "options": {
            "parallel_processing": true,
            "max_concurrent": 3
        }
    }
    """
    try:
        images = request.get("images", [])
        analyses = request.get("analyses", ["detection"])
        options = request.get("options", {})
        
        if not images:
            raise HTTPException(status_code=422, detail="No images provided")
        
        max_concurrent = min(options.get("max_concurrent", 3), 5)  # Cap at 5
        
        # Create analysis config
        config = AnalysisConfig(
            mode=AnalysisMode.COMPREHENSIVE if len(analyses) > 1 else AnalysisMode.DETECTION_ONLY,
            detection_config=DetectionConfig(
                engine=DetectionEngine(options.get("detection_engine", "YOLOV10N").upper()),
                confidence_threshold=options.get("confidence_threshold", 0.5),
                max_faces=options.get("max_faces", 10)
            ),
            recognition_config={
                "model": RecognitionModel(options.get("recognition_model", "ADAFACE").upper()),
                "threshold": options.get("recognition_threshold", 0.6)
            } if "recognition" in analyses else None,
            quality_level=QualityLevel.BALANCED
        )
        
        # Process images
        async def process_single_image(image_item):
            try:
                image_id = image_item.get("id", "unknown")
                image_b64 = image_item.get("data")
                
                if not image_b64:
                    return {
                        "image_id": image_id,
                        "success": False,
                        "error": "No image data provided"
                    }
                
                # Decode image
                image_data = base64.b64decode(image_b64)
                image_array = process_image_input(image_data)
                
                # Create gallery for recognition
                gallery = {}
                if "recognition" in analyses:
                    known_faces = request.get("known_faces", {})
                    for person_id, embedding_data in known_faces.items():
                        if isinstance(embedding_data, list):
                            gallery[person_id] = {
                                'name': person_id,
                                'embeddings': [np.array(embedding_data)]
                            }
                
                # Analyze image
                result = await analysis_service.analyze_faces(image_array, config, gallery)
                
                # Format result
                faces_data = []
                for i, face in enumerate(result.faces):
                    face_data = {
                        "face_id": f"face_{i+1:03d}",
                        "bbox": {
                            "x": int(face.detection.bbox.x1),
                            "y": int(face.detection.bbox.y1),
                            "width": int(face.detection.bbox.width),
                            "height": int(face.detection.bbox.height)
                        },
                        "confidence": float(face.detection.confidence)
                    }
                    
                    if "recognition" in analyses and face.recognition:
                        face_data["recognition"] = {
                            "unknown": len(face.recognition.matches) == 0,
                            "matches": [
                                {
                                    "person_id": match.identity_id,
                                    "similarity": float(match.similarity),
                                    "is_match": match.is_match
                                }
                                for match in face.recognition.matches
                            ]
                        }
                    
                    faces_data.append(face_data)
                
                return {
                    "image_id": image_id,
                    "success": True,
                    "faces_detected": len(result.faces),
                    "faces": faces_data,
                    "processing_time": result.processing_time
                }
            
            except Exception as e:
                return {
                    "image_id": image_item.get("id", "unknown"),
                    "success": False,
                    "error": str(e)
                }
        
        # Process with concurrency control
        if options.get("parallel_processing", True) and len(images) > 1:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(image_item):
                async with semaphore:
                    return await process_single_image(image_item)
            
            results = await asyncio.gather(*[
                process_with_semaphore(img) for img in images
            ])
        else:
            results = []
            for img in images:
                result = await process_single_image(img)
                results.append(result)
        
        # Aggregate statistics
        successful_results = [r for r in results if r["success"]]
        total_faces = sum(r.get("faces_detected", 0) for r in successful_results)
        total_processing_time = sum(r.get("processing_time", 0) for r in successful_results)
        
        response_data = {
            "success": True,
            "data": {
                "results": results,
                "summary": {
                    "images_processed": len(images),
                    "successful_analyses": len(successful_results),
                    "total_faces_detected": total_faces,
                    "failed_analyses": len(results) - len(successful_results)
                }
            },
            "meta": {
                "total_processing_time": total_processing_time,
                "timestamp": datetime.utcnow().isoformat(),
                "parallel_processing": options.get("parallel_processing", True),
                "max_concurrent": max_concurrent
            }
        }
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/switch-models")
async def switch_models(
    request: Dict[str, Any],
    analysis_service: FaceAnalysisService = Depends(get_face_analysis_service),
    vram_mgr: VRAMManager = Depends(get_vram_manager)
):
    """
    Switch detection and/or recognition models
    
    Request format:
    {
        "detection_engine": "yolov5s",
        "recognition_model": "arcface"
    }
    """
    try:
        detection_engine = request.get("detection_engine")
        recognition_model = request.get("recognition_model")
        
        results = {}
        
        # Switch detection engine
        if detection_engine:
            try:
                engine_enum = DetectionEngine(detection_engine.upper())
                await analysis_service.detection_service.switch_engine(engine_enum)
                results["detection_engine"] = {
                    "switched_to": detection_engine,
                    "success": True
                }
            except Exception as e:
                results["detection_engine"] = {
                    "requested": detection_engine,
                    "success": False,
                    "error": str(e)
                }
        
        # Switch recognition model
        if recognition_model:
            try:
                model_enum = RecognitionModel(recognition_model.upper())
                await analysis_service.recognition_service.switch_model(model_enum)
                results["recognition_model"] = {
                    "switched_to": recognition_model,
                    "success": True
                }
            except Exception as e:
                results["recognition_model"] = {
                    "requested": recognition_model,
                    "success": False,
                    "error": str(e)
                }
        
        # Get VRAM status
        vram_status = vram_mgr.get_memory_stats()
        
        response_data = {
            "success": True,
            "data": {
                "model_switches": results,
                "vram_status": {
                    "allocated_mb": vram_status["allocated_mb"],
                    "available_mb": vram_status["available_mb"],
                    "utilization_percent": vram_status["utilization_percent"]
                }
            },
            "meta": {
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model switching failed: {str(e)}")

@router.get("/status")
async def get_analysis_status(
    analysis_service: FaceAnalysisService = Depends(get_face_analysis_service),
    vram_mgr: VRAMManager = Depends(get_vram_manager)
):
    """Get current status of face analysis services"""
    try:
        # Get service statuses
        detection_status = analysis_service.detection_service.get_engine_info()
        recognition_status = analysis_service.recognition_service.get_model_info()
        vram_status = vram_mgr.get_memory_stats()
        
        # Get performance stats
        performance_stats = {}
        if hasattr(analysis_service.recognition_service, 'performance_stats'):
            perf = analysis_service.recognition_service.performance_stats
            performance_stats = {
                "total_embeddings_extracted": perf.total_embeddings_extracted,
                "total_comparisons": perf.total_comparisons,
                "average_extraction_time": perf.average_extraction_time,
                "average_comparison_time": perf.average_comparison_time
            }
        
        response_data = {
            "success": True,
            "data": {
                "services": {
                    "detection": {
                        "active_engine": detection_status.get("current_engine"),
                        "available_engines": detection_status.get("available_engines", []),
                        "engine_loaded": detection_status.get("model_loaded", False)
                    },
                    "recognition": {
                        "active_model": recognition_status.get("current_model"),
                        "available_models": recognition_status.get("available_models", []),
                        "model_loaded": recognition_status.get("model_loaded", False)
                    }
                },
                "performance": performance_stats,
                "vram": {
                    "allocated_mb": vram_status["allocated_mb"],
                    "available_mb": vram_status["available_mb"],
                    "utilization_percent": vram_status["utilization_percent"],
                    "allocations": vram_status.get("allocations", {})
                }
            },
            "meta": {
                "timestamp": datetime.utcnow().isoformat(),
                "service_version": "1.0.0"
            }
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Basic health check for face analysis service"""
    try:
        if face_analysis_service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "status": "unhealthy",
                    "message": "Face Analysis service not initialized"
                }
            )
        
        return JSONResponse(content={
            "success": True,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "face-analysis-api"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": "unhealthy",
                "error": str(e)
            }
        )
