# cSpell:disable
# mypy: ignore-errors
"""
จุดเริ่มต้นของแอปพลิเคชัน Face Detection Service
"""
import os
import sys
import logging

import uvicorn  # type: ignore
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel  # type: ignore

# เพิ่มโฟลเดอร์หลักไปยัง Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_services.common.vram_manager import VRAMManager
from src.ai_services.face_detection.face_detection_service import FaceDetectionService
from src.api.face_detection_api import router as face_detection_router, init_face_detection_api

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("face_detection_service.log")
    ]
)

logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """การตั้งค่าแอปพลิเคชัน"""
    app_name: str = "Face Detection Service"
    yolov9c_model_path: str = "model/face-detection/yolov9c-face-lindevs.onnx"
    yolov9e_model_path: str = "model/face-detection/yolov9e-face-lindevs.onnx"
    yolov11m_model_path: str = "model/face-detection/yolov11m-face.pt"
    max_usable_faces_yolov9: int = 8
    min_agreement_ratio: float = 0.7
    min_quality_threshold: int = 60
    conf_threshold: float = 0.15
    iou_threshold: float = 0.4
    img_size: int = 640


# สร้างแอปพลิเคชัน FastAPI
app = FastAPI(
    title="Face Detection Service API",
    description="บริการตรวจจับใบหน้าที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m",
    version="1.0.0"
)

# เพิ่ม CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตัวแปรสำหรับเก็บบริการตรวจจับใบหน้า
face_detection_service = None
vram_manager = None


@app.on_event("startup")
async def startup_event():
    """
    ฟังก์ชันที่ทำงานเมื่อเริ่มต้นแอปพลิเคชัน
    """
    global face_detection_service, vram_manager
    
    try:
        # โหลดการตั้งค่า
        settings = Settings()
        
        # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/face-detection", exist_ok=True)
          # ตั้งค่า VRAM Manager
        vram_manager_config = {
            "reserved_vram_mb": 512,
            "model_vram_estimates": {
                "yolov9c-face": 512 * 1024 * 1024,  # 512MB
                "yolov9e-face": 2048 * 1024 * 1024,  # 2GB สำหรับ YOLOv9e (เพิ่มจาก 1GB)
                "yolov11m-face": 2 * 1024 * 1024 * 1024,  # 2GB
            }
        }
        
        vram_manager = VRAMManager(vram_manager_config)
        
        # ตั้งค่าบริการตรวจจับใบหน้า
        face_detection_config = {
            "yolov9c_model_path": settings.yolov9c_model_path,
            "yolov9e_model_path": settings.yolov9e_model_path,
            "yolov11m_model_path": settings.yolov11m_model_path,
            "max_usable_faces_yolov9": settings.max_usable_faces_yolov9,
            "min_agreement_ratio": settings.min_agreement_ratio,
            "min_quality_threshold": settings.min_quality_threshold,
            "conf_threshold": settings.conf_threshold,
            "iou_threshold": settings.iou_threshold,
            "img_size": settings.img_size
        }
        
        face_detection_service = FaceDetectionService(vram_manager, face_detection_config)
        
        # โหลดโมเดล
        init_success = await face_detection_service.initialize()
        
        if not init_success:
            logger.error("ไม่สามารถโหลดโมเดลตรวจจับใบหน้าได้")
            # ไม่หยุดแอปพลิเคชัน แต่อาจจะมีข้อจำกัดในการใช้งาน
        
        # ตั้งค่า API
        init_face_detection_api(face_detection_service)
        
        logger.info(f"เริ่มต้น {settings.app_name} เรียบร้อยแล้ว")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการเริ่มต้นแอปพลิเคชัน: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    ฟังก์ชันที่ทำงานเมื่อปิดแอปพลิเคชัน
    """
    global face_detection_service
    
    try:
        if face_detection_service:
            await face_detection_service.cleanup()
        
        logger.info("ปิดแอปพลิเคชันเรียบร้อยแล้ว")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการปิดแอปพลิเคชัน: {e}")


# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์ก่อนที่จะ mount
os.makedirs("output", exist_ok=True)
os.makedirs("output/face-detection", exist_ok=True)

# เพิ่ม router
app.include_router(face_detection_router)

# เพิ่ม static files
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.get("/")
async def root():
    """
    หน้าแรกของ API
    """
    return {
        "service": "Face Detection Service",
        "status": "online",
        "documentation": "/docs",
        "models": ["YOLOv9c", "YOLOv9e", "YOLOv11m"]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
