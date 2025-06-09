# cSpell:disable
# mypy: ignore-errors
# pylint: disable=all
"""
API สำหรับบริการตรวจจับใบหน้า (Face Detection API)
"""
import os
import time
import uuid
import base64
import logging
from typing import Dict, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query  # type: ignore
from fastapi.responses import JSONResponse, FileResponse  # type: ignore
import cv2
import numpy as np

from ..ai_services.face_detection.face_detection_service import FaceDetectionService  # type: ignore
from ..ai_services.face_detection.utils import save_detection_image  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/face-detection",
    tags=["face-detection"],
    responses={404: {"description": "Not found"}},
)

# เก็บผลลัพธ์การตรวจจับล่าสุด
detection_cache: Dict[str, Dict[str, Any]] = {}


def cleanup_old_results():
    """ลบผลลัพธ์เก่าเกิน 1 ชั่วโมง"""
    current_time = time.time()
    expired_keys = []
    
    for key, value in detection_cache.items():
        if current_time - value["timestamp"] > 3600:  # 1 ชั่วโมง
            expired_keys.append(key)
            
            # ลบไฟล์รูปภาพด้วย
            if "output_image_path" in value and os.path.exists(value["output_image_path"]):
                try:
                    os.remove(value["output_image_path"])
                except Exception as e:
                    logger.error(f"ไม่สามารถลบไฟล์ {value['output_image_path']}: {e}")
    
    for key in expired_keys:
        del detection_cache[key]


def init_face_detection_api(face_detection_service: FaceDetectionService):
    """
    ตั้งค่า API และเชื่อมต่อกับบริการตรวจจับใบหน้า
    
    Args:
        face_detection_service: บริการตรวจจับใบหน้า
    """
    router.face_detection_service = face_detection_service


@router.post("/detect")
async def detect_faces(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    model: Optional[str] = Query(None, enum=["yolov9c", "yolov9e", "yolov11m", "auto"]),
    conf_threshold: Optional[float] = Query(0.15, gt=0.0, lt=1.0),
    iou_threshold: Optional[float] = Query(0.4, gt=0.0, lt=1.0),
    return_image: bool = Query(False),
):
    """
    ตรวจจับใบหน้าในรูปภาพ
    
    Args:
        file: ไฟล์รูปภาพ
        image_base64: รูปภาพในรูปแบบ base64
        model: โมเดลที่ต้องการใช้
        conf_threshold: ระดับความมั่นใจขั้นต่ำ
        iou_threshold: ค่า IoU threshold สำหรับ NMS
        return_image: ส่งคืนรูปภาพที่มีการวาดกรอบใบหน้าหรือไม่
        
    Returns:
        ผลลัพธ์การตรวจจับใบหน้า
    """
    try:
        # ตรวจสอบว่ามีรูปภาพหรือไม่
        if file is None and image_base64 is None:
            raise HTTPException(status_code=400, detail="ต้องระบุไฟล์รูปภาพหรือรูปภาพแบบ base64")
        
        # แปลงรูปภาพ
        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="ไม่สามารถอ่านไฟล์รูปภาพได้")
        else:
            try:
                # ตัดส่วน "data:image/jpeg;base64," ออกถ้ามี
                image_base64_str = image_base64 or ""
                if "," in image_base64_str:
                    image_base64_str = image_base64_str.split(",")[1]
                
                img_data = base64.b64decode(image_base64_str)
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise HTTPException(status_code=400, detail="ไม่สามารถอ่านรูปภาพ base64 ได้")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"รูปแบบ base64 ไม่ถูกต้อง: {str(e)}")
        
        # ตรวจจับใบหน้า
        result = await router.face_detection_service.detect_faces(
            image, model, conf_threshold, iou_threshold
        )
        
        # สร้าง ID สำหรับผลลัพธ์
        result_id = str(uuid.uuid4())
        
        # แปลงผลลัพธ์เป็น JSON
        result_json = result.to_dict()
        
        # ถ้าต้องการรูปภาพที่มีการวาดกรอบใบหน้า
        if return_image:
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            output_dir = "output/face-detection"
            os.makedirs(output_dir, exist_ok=True)
            
            # สร้างชื่อไฟล์
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"face_detection_{timestamp}_{result_id[:8]}.jpg"
            
            # วาดกรอบใบหน้าและบันทึกรูปภาพ
            output_path = save_detection_image(image, result.faces, output_dir, output_filename)
            
            # เพิ่ม URL สำหรับดาวน์โหลดรูปภาพ
            result_json["output_image_url"] = f"/api/face-detection/result-image/{result_id}"
            
            # เก็บข้อมูลในแคช
            detection_cache[result_id] = {
                "result": result_json,
                "output_image_path": output_path,
                "timestamp": time.time()
            }
        else:
            # เก็บเฉพาะผลลัพธ์
            detection_cache[result_id] = {
                "result": result_json,
                "timestamp": time.time()
            }
        
        # ตั้งเวลาทำความสะอาดผลลัพธ์เก่า
        background_tasks.add_task(cleanup_old_results)
        
        # เพิ่ม ID สำหรับดึงผลลัพธ์ในภายหลัง
        result_json["result_id"] = result_id
        
        return JSONResponse(content=result_json)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการตรวจจับใบหน้า: {e}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการตรวจจับใบหน้า: {str(e)}")


@router.get("/result/{result_id}")
async def get_detection_result(result_id: str):
    """
    ดึงผลลัพธ์การตรวจจับใบหน้าจาก ID
    
    Args:
        result_id: ID ของผลลัพธ์
        
    Returns:
        ผลลัพธ์การตรวจจับใบหน้า
    """
    if result_id not in detection_cache:
        raise HTTPException(status_code=404, detail="ไม่พบผลลัพธ์สำหรับ ID นี้")
    
    return JSONResponse(content=detection_cache[result_id]["result"])


@router.get("/result-image/{result_id}")
async def get_result_image(result_id: str):
    """
    ดึงรูปภาพผลลัพธ์การตรวจจับใบหน้าจาก ID
    
    Args:
        result_id: ID ของผลลัพธ์
        
    Returns:
        รูปภาพที่มีการวาดกรอบใบหน้า
    """
    if result_id not in detection_cache or "output_image_path" not in detection_cache[result_id]:
        raise HTTPException(status_code=404, detail="ไม่พบรูปภาพผลลัพธ์สำหรับ ID นี้")
    
    image_path = detection_cache[result_id]["output_image_path"]
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์รูปภาพ")
    
    return FileResponse(image_path)


@router.get("/status")
async def get_service_status():
    """
    ดูสถานะของบริการตรวจจับใบหน้า
    
    Returns:
        ข้อมูลสถานะบริการ
    """
    try:
        service_info = await router.face_detection_service.get_service_info()
        
        return {
            "status": "online",
            "service_info": service_info,
            "cache_size": len(detection_cache)
        }
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงสถานะบริการ: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
