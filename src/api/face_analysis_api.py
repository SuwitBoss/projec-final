"""
Face Analysis API - Integrated Face Detection + Recognition Endpoints
Comprehensive face analysis combining detection and recognition services
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any, Union
import numpy as np # For image processing if needed, and for type hints
from datetime import datetime # For timestamps
import base64 # For base64 image processing in analyze_faces_base64

from ..ai_services.face_analysis.models import (
    AnalysisConfig,
    AnalysisMode,  # Added
    QualityLevel   # Added
)
from ..ai_services.face_detection.models import ( # Added this import block
    DetectionConfig,
    DetectionEngine
)
from ..ai_services.face_recognition.models import ( # Added this import block
    RecognitionModel
)
from ..ai_services.face_analysis.face_analysis_service import FaceAnalysisService
# FaceDetectionService is imported but reported as unused in the last check, let's keep it for now as /detect endpoint might need it.
from ..ai_services.face_detection.face_detection_service import FaceDetectionService
from ..ai_services.common.vram_manager import VRAMManager # For model switching logic
from ..utils.image_utils import process_image_input # process_image_input is used

router = APIRouter(prefix="/api/face-analysis", tags=["face-analysis"])

# Global service references (will be injected from main.py)
face_analysis_service: Optional[FaceAnalysisService] = None
face_detection_service: Optional[FaceDetectionService] = None # Corrected type hint
vram_manager: Optional[VRAMManager] = None

def get_face_analysis_service() -> FaceAnalysisService:
    """Dependency to get face analysis service"""
    if face_analysis_service is None:
        raise HTTPException(status_code=503, detail="Face analysis service not available or not initialized.") # Added detail
    return face_analysis_service

def get_face_detection_service() -> FaceDetectionService: # Corrected type hint
    """Dependency to get face detection service directly"""
    if face_detection_service is None:
        raise HTTPException(status_code=503, detail="Face detection service not available")
    return face_detection_service

def get_vram_manager() -> VRAMManager:
    """Dependency to get VRAM manager"""
    if vram_manager is None:
        raise HTTPException(status_code=503, detail="VRAM manager not initialized")
    return vram_manager

@router.get("/health")
async def health_check():
    """Health check endpoint for face analysis service"""
    try:
        # Check if services are available
        detection_service_available = face_analysis_service is not None or face_detection_service is not None
        vram_mgr_available = vram_manager is not None
        
        current_status = "healthy"
        if not detection_service_available:
            current_status = "degraded (detection service unavailable)"
        elif not vram_mgr_available:
            current_status = "degraded (VRAM manager unavailable)"

        models_status_dict = {}
        # Determine which detection service instance to check
        active_detection_svc = None
        if face_analysis_service and hasattr(face_analysis_service, 'face_detection_service'):
            active_detection_svc = face_analysis_service.face_detection_service
        elif face_detection_service:
            active_detection_svc = face_detection_service
        
        if active_detection_svc and hasattr(active_detection_svc, 'models'):
            models_ref = active_detection_svc.models
            models_status_dict = {
                "yolov9c": "loaded" if "yolov9c" in models_ref and models_ref["yolov9c"].model_loaded else "not_loaded",
                "yolov9e": "loaded" if "yolov9e" in models_ref and models_ref["yolov9e"].model_loaded else "not_loaded", 
                "yolov11m": "loaded" if "yolov11m" in models_ref and models_ref["yolov11m"].model_loaded else "not_loaded"
            }
            # Add enhanced detector status if applicable
            if hasattr(active_detection_svc, 'use_enhanced_detector') and active_detection_svc.use_enhanced_detector:
                 models_status_dict['enhanced_detector'] = "loaded" if 'enhanced' in models_ref and models_ref['enhanced'].model_loaded else "not_loaded"
        
        return {
            "status": current_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "face_detection_direct": "available" if face_detection_service is not None else "unavailable",
                "face_analysis_wrapper": "available" if face_analysis_service is not None else "unavailable",
                "vram_manager": "available" if vram_mgr_available else "unavailable"
            },
            "models": models_status_dict,
            "version": "1.0.2", # Incremented version
            "endpoints": [
                "/api/face-analysis/health",
                "/api/face-analysis/detect", 
                "/api/face-analysis/analyze"
                # Add other relevant endpoints if they exist
            ]
        }
    except Exception as e:
        # Consider logging the exception e
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "1.0.2"
        }

@router.post("/detect")
async def detect_faces(
    image: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(default=0.5),
    min_face_size: Optional[int] = Form(default=60),
    max_faces: Optional[int] = Form(default=10), # Parameter name in FaceDetectionService
    return_landmarks: Optional[bool] = Form(default=False) # Parameter name in FaceDetectionService
):
    """
    Face detection only endpoint for real-time use
    
    Args:
        image: Input image file
        confidence_threshold: Minimum confidence for detection (0.1-0.9)
        min_face_size: Minimum face size in pixels (width/height for square)
        max_faces: Maximum number of faces to detect
        return_landmarks: Whether to return facial landmarks
    """
    try:
        image_data = await image.read()
        image_array = process_image_input(image_data) # Ensure this returns np.ndarray
        
        active_detection_service = None
        if face_analysis_service and hasattr(face_analysis_service, 'face_detection_service'):
            active_detection_service = face_analysis_service.face_detection_service
        elif face_detection_service:
            active_detection_service = face_detection_service
        
        if active_detection_service:
            # Parameters for FaceDetectionService.detect_faces:
            # image_input, model_name, conf_threshold, iou_threshold, min_face_size (tuple), max_faces, return_landmarks
            result = await active_detection_service.detect_faces(
                image_input=image_array,
                model_name="yolov9c", # Default or make configurable
                conf_threshold=confidence_threshold,
                max_faces=max_faces,
                min_face_size=(min_face_size, min_face_size) if min_face_size else None,
                return_landmarks=return_landmarks
            )
            
            # DEBUG: Log the detection result
            print(f"🔍 DEBUG - Detection result: faces={len(result.faces)}, model_used={result.model_used}")
            # (Keep other debug prints if needed)

            faces_response_list = []
            for i, face_obj in enumerate(result.faces):
                face_data = {
                    "face_id": f"face_{i+1:03d}",
                    "bbox": {
                        "x": int(face_obj.bbox.x1),
                        "y": int(face_obj.bbox.y1),
                        "width": int(face_obj.bbox.width),
                        "height": int(face_obj.bbox.height)
                    },
                    "confidence": float(face_obj.bbox.confidence),
                    "quality_score": float(face_obj.quality_score) if face_obj.quality_score is not None else 0.0,
                    "engine_used": face_obj.model_used or "unknown_yolo"
                }
                if return_landmarks and hasattr(face_obj, 'landmarks') and face_obj.landmarks:
                    face_data["landmarks"] = {
                        "points": [[float(p[0]), float(p[1])] for p in face_obj.landmarks.points],
                        "type": face_obj.landmarks.landmark_type
                    }
                faces_response_list.append(face_data)
            
            return {
                "success": True,
                "data": {
                    "faces": faces_response_list,
                    "total_faces": len(faces_response_list),
                    "processing_time": result.total_processing_time, # Assuming this is in ms or a suitable unit
                    "engine_used": result.model_used,
                    "fallback_used": result.fallback_used,
                    "image_shape": result.image_shape
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Face detection service not available")
            
    except Exception as e:
        # print(f"Error in /detect: {str(e)}") # For debugging
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

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
        # Ensure DetectionEngine and RecognitionModel enums are correctly used
        current_detection_engine_str = detection_engine.upper()
        if not hasattr(DetectionEngine, current_detection_engine_str):
            # Handle invalid engine string, perhaps default or raise error
            current_detection_engine_str = "YOLOV10N" # Defaulting
            
        current_recognition_model_str = recognition_model.upper()
        if not hasattr(RecognitionModel, current_recognition_model_str):
            # Handle invalid model string
            current_recognition_model_str = "ADAFACE" # Defaulting

        config = AnalysisConfig(
            mode=AnalysisMode.COMPREHENSIVE if len(analysis_types) > 1 else AnalysisMode.DETECTION_ONLY,
            detection_config=DetectionConfig(
                engine=DetectionEngine[current_detection_engine_str],
                confidence_threshold=confidence_threshold,
                max_faces=max_faces,
                return_landmarks=True # Often needed for analysis
            ),
            recognition_config={
                "model": RecognitionModel[current_recognition_model_str],
                "threshold": 0.6, # Example, make configurable if needed
                "return_embeddings": return_embeddings
            } if "recognition" in analysis_types else None, # Ensure this structure is fine
            quality_level=QualityLevel.BALANCED
        )
        
        gallery = None # Initialize gallery
        if "recognition" in analysis_types:
            gallery = {} # Placeholder for actual gallery loading logic
        
        result = await analysis_service.analyze_faces(image_array, config, gallery)
        
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
                "timestamp": datetime.utcnow().isoformat(), # Use utcnow for consistency
                "analyses_performed": analysis_types,
                "models_used": {
                    "detection": config.detection_config.engine.value if config.detection_config else None,
                    "recognition": config.recognition_config.get("model").value if config.recognition_config and config.recognition_config.get("model") else None
                }
            }
        }
        
        for i, face_obj in enumerate(result.faces): # Renamed 'face' to 'face_obj'
            face_data = {
                "face_id": f"face_{i+1:03d}",
                "bbox": {
                    "x": int(face_obj.detection.bbox.x1),
                    "y": int(face_obj.detection.bbox.y1),
                    "width": int(face_obj.detection.bbox.width),
                    "height": int(face_obj.detection.bbox.height)
                },
                "confidence": float(face_obj.detection.confidence),
                "quality_score": float(face_obj.quality.overall_quality) if face_obj.quality else 0.0
            }
            
            if "detection" in analysis_types:
                face_data["detection"] = {
                    "confidence": float(face_obj.detection.confidence),
                    "engine_used": face_obj.detection.engine_used,
                    "quality_score": float(face_obj.quality.overall_quality) if face_obj.quality else 0.0
                }
                if face_obj.detection.landmarks:
                    face_data["detection"]["landmarks"] = {
                        "points": [[float(p[0]), float(p[1])] for p in face_obj.detection.landmarks.points],
                        "type": face_obj.detection.landmarks.landmark_type
                        }
            
            if "recognition" in analysis_types and face_obj.recognition:
                face_data["recognition"] = {
                    "unknown": len(face_obj.recognition.matches) == 0,
                    "confidence": float(face_obj.recognition.embedding_quality) if face_obj.recognition.embedding_quality else 0.0,
                    "matches": []
                }
                if face_obj.recognition.matches:
                     face_data["recognition"]["matches"] = [
                        {
                            "identity_id": match.identity_id,
                            "similarity": float(match.similarity),
                            "is_match": match.is_match # Assuming this attribute exists
                        } for match in face_obj.recognition.matches
                    ]
                if return_embeddings and face_obj.recognition.embedding is not None:
                    face_data["recognition"]["embedding"] = face_obj.recognition.embedding.tolist()
            
            response_data["data"]["faces"].append(face_data)
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        # print(f"Error in /analyze: {str(e)}") # For debugging
        return JSONResponse(content={"success": False, "error": str(e)})

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
        except Exception as e_decode: # More specific exception handling
            raise HTTPException(status_code=422, detail=f"Invalid image data: {str(e_decode)}")
        
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
        for i, face_obj in enumerate(result.faces):
            face_data = {
                "face_id": f"face_{i+1:03d}",
                "bbox": {
                    "x": int(face_obj.detection.bbox.x1),
                    "y": int(face_obj.detection.bbox.y1), 
                    "width": int(face_obj.detection.bbox.width),
                    "height": int(face_obj.detection.bbox.height)
                },
                "confidence": float(face_obj.detection.confidence)
            }
            
            if "recognition" in analyses and face_obj.recognition:
                face_data["recognition"] = {
                    "unknown": len(face_obj.recognition.matches) == 0,
                    "matches": [
                        {
                            "person_id": match.identity_id,
                            "similarity": float(match.similarity),
                            "is_match": match.is_match
                        }
                        for match in face_obj.recognition.matches
                    ]
                }
                
                if options.get("return_embeddings") and face_obj.recognition.embedding:
                    face_data["recognition"]["embedding"] = face_obj.recognition.embedding.tolist()
            
            response_data["data"]["faces"].append(face_data)
        
        return JSONResponse(content=response_data)
    
    except HTTPException: # Important to re-raise HTTPExceptions
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
    """Batch analysis of multiple images"""
    try:
        images = request.get("images", [])
        # analyses = request.get("analyses", ["detection"]) # Not used in current placeholder
        # options = request.get("options", {}) # Not used in current placeholder
        
        if not images:
            raise HTTPException(status_code=422, detail="No images provided for batch analysis")
        
        # max_concurrent = min(options.get("max_concurrent", 3), 5)  # Cap at 5 # Not used
        
        # Create analysis config # Not used
        # Placeholder implementation for the rest of the try block
        # In a real implementation, you would iterate through images, call analysis_service,
        # and aggregate results, possibly using asyncio.gather for concurrency.
        return JSONResponse(content={"message": "Batch analysis not fully implemented", "status": "placeholder", "received_images": len(images)})

    except HTTPException:
         raise # Re-raise the HTTPException to be handled by FastAPI's default error handling
    except Exception as e:
        # It's good practice to log the exception here for server-side debugging
        # import logging
        # logging.error(f"Error in batch_analyze_faces: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": f"Batch analysis failed: {str(e)}"},
            status_code=500
        )

@router.post("/switch-models")
async def switch_models(
    request_data: Dict[str, Any], 
    # vram_manager_dep: VRAMManager = Depends(get_vram_manager) # Optional: if direct VRAM ops needed here
):
    """
    Endpoint to switch active AI models for detection or other services.
    Example request:
    {
        "service": "face_detection",
        "model_name": "yolov9e" 
    }
    """
    service_to_switch = request_data.get("service")
    new_model_name = request_data.get("model_name")

    if not service_to_switch or not new_model_name:
        raise HTTPException(status_code=422, detail="Missing 'service' or 'model_name' in request.")

    # Access global service variables. In a larger app, services might be managed by a central registry.
    global face_detection_service
    global face_analysis_service 
    # Add other services if they also need model switching capabilities

    if service_to_switch == "face_detection":
        active_detection_service_to_configure = None
        service_name_for_message = ""

        # Prioritize face_analysis_service if it wraps the detection service
        if face_analysis_service and hasattr(face_analysis_service, 'face_detection_service'):
            active_detection_service_to_configure = face_analysis_service.face_detection_service
            service_name_for_message = "face_analysis_service's detection component"
        elif face_detection_service:
            active_detection_service_to_configure = face_detection_service
            service_name_for_message = "direct face_detection_service"
        
        if active_detection_service_to_configure and hasattr(active_detection_service_to_configure, 'switch_active_model'):
            try:
                # This conceptual 'switch_active_model' method would need to be implemented
                # in the FaceDetectionService class. It should handle loading the new model
                # (if not already loaded) and setting it as the active one, potentially
                # using the VRAMManager to manage GPU memory.
                # await active_detection_service_to_configure.switch_active_model(new_model_name)
                
                # Placeholder: Simulate a check and response
                # Actual model validation should be robust, checking against available model files/configs.
                if new_model_name not in ["yolov9c", "yolov9e", "yolov11m", "yolov10n", "yolov5s", "mediapipe", "insightface"]: # Add all valid models
                    return JSONResponse(
                        content={"success": False, "message": f"Model '{new_model_name}' is not a recognized or supported detection model."},
                        status_code=400
                    )
                
                # Simulate successful switch for now.
                # In reality, this would involve re-configuring the service instance.
                # For example: active_detection_service_to_configure.current_model_key = new_model_name
                # Or: await active_detection_service_to_configure.load_model(new_model_name, force_reload=True)

                return JSONResponse(content={
                    "success": True, 
                    "message": f"Request to switch {service_name_for_message} to model '{new_model_name}' acknowledged. "
                               f"This is a placeholder; actual model switching logic needs full implementation in the service."
                })
            except Exception as e_switch:
                # import logging
                # logging.error(f"Error switching model for {service_name_for_message}: {e_switch}", exc_info=True)
                return JSONResponse(
                    content={"success": False, "error": f"Error during model switch attempt for {service_name_for_message}: {str(e_switch)}"},
                    status_code=500
                )
        elif not active_detection_service_to_configure:
             return JSONResponse(
                content={"success": False, "message": "No active face detection service available for model switching."},
                status_code=503 # Service Unavailable
            )
        else: # Service exists but no switch_active_model method (or equivalent logic)
            return JSONResponse(
                content={"success": False, "message": f"Model switching for '{service_name_for_message}' is not supported by the current service implementation."},
                status_code=501 # Not Implemented
            )
    # Add elif blocks for other services like "face_recognition", "antispoofing" etc.
    # elif service_to_switch == "face_recognition":
    #    ...
    else:
        return JSONResponse(
            content={"success": False, "message": f"Service '{service_to_switch}' is not recognized or not supported for model switching."},
            status_code=400 # Bad Request
        )

# Ensure the file ends with a newline.
