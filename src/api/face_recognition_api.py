"""
Face Recognition API Endpoints
API สำหรับระบบจดจำใบหน้า
"""

import io
import base64
import logging
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
import cv2

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..ai_services.face_recognition.face_recognition_service import FaceRecognitionService
from ..ai_services.face_recognition.models import FaceGallery

# ตั้งค่า logger
logger = logging.getLogger(__name__)

# สร้าง router
router = APIRouter(prefix="/api/face-recognition", tags=["face-recognition"])

# Global service instance
face_recognition_service: Optional[FaceRecognitionService] = None


# Pydantic models สำหรับ request/response
class EmbeddingRequest(BaseModel):
    image_base64: str
    model_name: Optional[str] = None


class ComparisonRequest(BaseModel):
    image1_base64: str
    image2_base64: str
    model_name: Optional[str] = None


class RecognitionRequest(BaseModel):
    image_base64: str
    gallery: Dict[str, Any]  # FaceGallery format
    model_name: Optional[str] = None
    top_k: int = 5


class EmbeddingResponse(BaseModel):
    success: bool
    embedding: Optional[List[float]] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    face_quality: Optional[float] = None
    error: Optional[str] = None


class ComparisonResponse(BaseModel):
    success: bool
    similarity: Optional[float] = None
    confidence: Optional[float] = None
    is_same_person: Optional[bool] = None
    threshold_used: Optional[float] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    quality: Optional[str] = None
    error: Optional[str] = None


class RecognitionResponse(BaseModel):
    success: bool
    matches: Optional[List[Dict[str, Any]]] = None
    best_match: Optional[Dict[str, Any]] = None
    embedding: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    has_match: Optional[bool] = None
    identity: Optional[str] = None
    error: Optional[str] = None


def init_face_recognition_api(service: FaceRecognitionService):
    """เริ่มต้น Face Recognition API"""
    global face_recognition_service
    face_recognition_service = service
    logger.info("Face Recognition API initialized")


def _decode_base64_image(image_base64: str) -> np.ndarray:
    """แปลง base64 เป็น numpy array"""
    try:
        # ลบ header ถ้ามี
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        
        # แปลงเป็น PIL Image
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # แปลงเป็น RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # แปลงเป็น numpy array
        image_array = np.array(image_pil)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")


def _load_image_from_upload(file: UploadFile) -> np.ndarray:
    """โหลดรูปจาก UploadFile"""
    try:
        # อ่านไฟล์
        contents = file.file.read()
        
        # แปลงเป็น PIL Image
        image_pil = Image.open(io.BytesIO(contents))
        
        # แปลงเป็น RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # แปลงเป็น numpy array
        image_array = np.array(image_pil)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Failed to load image from upload: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


@router.get("/models")
async def get_available_models():
    """ดึงรายการโมเดลที่มีอยู่"""
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=503, detail="Face Recognition service not available")
        
        models_info = face_recognition_service.model_selector.get_performance_comparison()
        return JSONResponse(content={
            "success": True,
            "models": models_info
        })
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_performance_stats():
    """ดึงสถิติประสิทธิภาพ"""
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=503, detail="Face Recognition service not available")
        
        stats = face_recognition_service.get_performance_stats()
        return JSONResponse(content={
            "success": True,
            "stats": stats.to_dict() if stats else None
        })
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-embedding")
async def extract_embedding_endpoint(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    model_name: Optional[str] = Query(None, description="Model to use (adaface, arcface, facenet)")
) -> EmbeddingResponse:
    """
    สกัด face embedding จากรูปใบหน้า
    """
    try:
        if not face_recognition_service:
            return EmbeddingResponse(
                success=False,
                error="Face Recognition service not available"
            )
        
        # โหลดรูปภาพ
        if file:
            image = _load_image_from_upload(file)
        elif image_base64:
            image = _decode_base64_image(image_base64)
        else:
            return EmbeddingResponse(
                success=False,
                error="No image provided"
            )
        
        # สกัด embedding
        result = await face_recognition_service.extract_embedding(image, model_name)
        
        return EmbeddingResponse(
            success=True,
            embedding=result.embedding.tolist(),
            confidence=result.confidence,
            model_used=result.model_used,
            processing_time=result.processing_time,
            face_quality=result.face_quality
        )
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return EmbeddingResponse(
            success=False,
            error=str(e)
        )


@router.post("/compare-faces")
async def compare_faces_endpoint(
    file1: Optional[UploadFile] = File(None),
    file2: Optional[UploadFile] = File(None),
    image1_base64: Optional[str] = Form(None),
    image2_base64: Optional[str] = Form(None),
    model_name: Optional[str] = Query(None, description="Model to use")
) -> ComparisonResponse:
    """
    เปรียบเทียบใบหน้า 2 ใบ
    """
    try:
        if not face_recognition_service:
            return ComparisonResponse(
                success=False,
                error="Face Recognition service not available"
            )
        
        # โหลดรูปภาพที่ 1
        if file1:
            image1 = _load_image_from_upload(file1)
        elif image1_base64:
            image1 = _decode_base64_image(image1_base64)
        else:
            return ComparisonResponse(
                success=False,
                error="No first image provided"
            )
        
        # โหลดรูปภาพที่ 2
        if file2:
            image2 = _load_image_from_upload(file2)
        elif image2_base64:
            image2 = _decode_base64_image(image2_base64)
        else:
            return ComparisonResponse(
                success=False,
                error="No second image provided"
            )
        
        # สกัด embeddings
        embedding1 = await face_recognition_service.extract_embedding(image1, model_name)
        embedding2 = await face_recognition_service.extract_embedding(image2, model_name)
        
        # เปรียบเทียบ
        comparison = face_recognition_service.compare_faces(
            embedding1.embedding,
            embedding2.embedding,
            embedding1.model_used
        )
        
        return ComparisonResponse(
            success=True,
            similarity=comparison.similarity,
            confidence=comparison.confidence,
            is_same_person=comparison.is_same_person,
            threshold_used=comparison.threshold_used,
            model_used=comparison.model_used,
            processing_time=comparison.processing_time,
            quality=comparison.quality.value
        )
        
    except Exception as e:
        logger.error(f"Face comparison failed: {e}")
        return ComparisonResponse(
            success=False,
            error=str(e)
        )


@router.post("/recognize-face")
async def recognize_face_endpoint(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    gallery: str = Form(..., description="JSON string of face gallery"),
    model_name: Optional[str] = Query(None, description="Model to use"),
    top_k: int = Query(5, description="Number of top matches to return")
) -> RecognitionResponse:
    """
    จดจำใบหน้าจากฐานข้อมูล
    """
    try:
        if not face_recognition_service:
            return RecognitionResponse(
                success=False,
                error="Face Recognition service not available"
            )
        
        # โหลดรูปภาพ
        if file:
            image = _load_image_from_upload(file)
        elif image_base64:
            image = _decode_base64_image(image_base64)
        else:
            return RecognitionResponse(
                success=False,
                error="No image provided"
            )
        
        # Parse gallery
        import json
        try:
            gallery_dict = json.loads(gallery)
            # แปลง list เป็น numpy arrays
            for identity_id, data in gallery_dict.items():
                if 'embeddings' in data:
                    embeddings = []
                    for emb in data['embeddings']:
                        if isinstance(emb, list):
                            embeddings.append(np.array(emb, dtype=np.float32))
                        else:
                            embeddings.append(emb)
                    gallery_dict[identity_id]['embeddings'] = embeddings
        except json.JSONDecodeError as e:
            return RecognitionResponse(
                success=False,
                error=f"Invalid gallery format: {e}"
            )
        
        # จดจำใบหน้า
        result = await face_recognition_service.recognize_face(
            image, gallery_dict, model_name, top_k
        )
        
        return RecognitionResponse(
            success=True,
            matches=[match.to_dict() for match in result.matches],
            best_match=result.best_match.to_dict() if result.best_match else None,
            embedding=result.face_embedding.to_dict(),
            processing_time=result.processing_time,
            model_used=result.model_used,
            has_match=result.has_match,
            identity=result.identity
        )
        
    except Exception as e:
        logger.error(f"Face recognition failed: {e}")
        return RecognitionResponse(
            success=False,
            error=str(e)
        )


@router.post("/batch-extract-embeddings")
async def batch_extract_embeddings_endpoint(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Query(None, description="Model to use")
):
    """
    สกัด embeddings หลายรูปพร้อมกัน
    """
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=503, detail="Face Recognition service not available")
        
        # โหลดรูปภาพทั้งหมด
        images = []
        for file in files:
            image = _load_image_from_upload(file)
            images.append(image)
        
        # สกัด embeddings
        embeddings = await face_recognition_service.batch_extract_embeddings(images, model_name)
        
        # แปลงผลลัพธ์
        results = []
        for i, embedding in enumerate(embeddings):
            results.append({
                'index': i,
                'filename': files[i].filename,
                'embedding': embedding.to_dict()
            })
        
        return JSONResponse(content={
            "success": True,
            "total_images": len(files),
            "successful_extractions": len(embeddings),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Batch embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch-model")
async def switch_model_endpoint(
    model_name: str = Body(..., embed=True, description="Model name to switch to")
):
    """
    เปลี่ยนโมเดลที่ใช้งาน
    """
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=503, detail="Face Recognition service not available")
        
        success = await face_recognition_service.switch_model(model_name)
        
        return JSONResponse(content={
            "success": success,
            "current_model": face_recognition_service.current_model,
            "message": f"Switched to {model_name}" if success else f"Failed to switch to {model_name}"
        })
        
    except Exception as e:
        logger.error(f"Model switching failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# JSON API endpoints (alternative to form-data)
@router.post("/extract-embedding-json")
async def extract_embedding_json_endpoint(request: EmbeddingRequest) -> EmbeddingResponse:
    """สกัด embedding (JSON API)"""
    try:
        if not face_recognition_service:
            return EmbeddingResponse(success=False, error="Service not available")
        
        image = _decode_base64_image(request.image_base64)
        result = await face_recognition_service.extract_embedding(image, request.model_name)
        
        return EmbeddingResponse(
            success=True,
            embedding=result.embedding.tolist(),
            confidence=result.confidence,
            model_used=result.model_used,
            processing_time=result.processing_time,
            face_quality=result.face_quality
        )
        
    except Exception as e:
        logger.error(f"JSON embedding extraction failed: {e}")
        return EmbeddingResponse(success=False, error=str(e))


@router.post("/compare-faces-json")
async def compare_faces_json_endpoint(request: ComparisonRequest) -> ComparisonResponse:
    """เปรียบเทียบใบหน้า (JSON API)"""
    try:
        if not face_recognition_service:
            return ComparisonResponse(success=False, error="Service not available")
        
        image1 = _decode_base64_image(request.image1_base64)
        image2 = _decode_base64_image(request.image2_base64)
        
        embedding1 = await face_recognition_service.extract_embedding(image1, request.model_name)
        embedding2 = await face_recognition_service.extract_embedding(image2, request.model_name)
        
        comparison = face_recognition_service.compare_faces(
            embedding1.embedding, embedding2.embedding, embedding1.model_used
        )
        
        return ComparisonResponse(
            success=True,
            similarity=comparison.similarity,
            confidence=comparison.confidence,
            is_same_person=comparison.is_same_person,
            threshold_used=comparison.threshold_used,
            model_used=comparison.model_used,
            processing_time=comparison.processing_time,
            quality=comparison.quality.value
        )
        
    except Exception as e:
        logger.error(f"JSON face comparison failed: {e}")
        return ComparisonResponse(success=False, error=str(e))


@router.post("/recognize-face-json")
async def recognize_face_json_endpoint(request: RecognitionRequest) -> RecognitionResponse:
    """จดจำใบหน้า (JSON API)"""
    try:
        if not face_recognition_service:
            return RecognitionResponse(success=False, error="Service not available")
        
        image = _decode_base64_image(request.image_base64)
        
        # แปลง gallery
        gallery_dict = request.gallery.copy()
        for identity_id, data in gallery_dict.items():
            if 'embeddings' in data:
                embeddings = []
                for emb in data['embeddings']:
                    if isinstance(emb, list):
                        embeddings.append(np.array(emb, dtype=np.float32))
                    else:
                        embeddings.append(emb)
                gallery_dict[identity_id]['embeddings'] = embeddings
        
        result = await face_recognition_service.recognize_face(
            image, gallery_dict, request.model_name, request.top_k
        )
        
        return RecognitionResponse(
            success=True,
            matches=[match.to_dict() for match in result.matches],
            best_match=result.best_match.to_dict() if result.best_match else None,
            embedding=result.face_embedding.to_dict(),
            processing_time=result.processing_time,
            model_used=result.model_used,
            has_match=result.has_match,
            identity=result.identity
        )
        
    except Exception as e:
        logger.error(f"JSON face recognition failed: {e}")
        return RecognitionResponse(success=False, error=str(e))
