# cSpell:disable
# mypy: ignore-errors
"""
จุดเริ่มต้นของแอปพลิเคชัน Face Detection Service
"""
import os
import sys
import logging

# เพิ่มเส้นทางโครงการเข้าไปใน sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# นำเข้า modules
from src.ai_services.common.vram_manager import VRAMManager  # type: ignore
from src.ai_services.face_detection.face_detection_service import FaceDetectionService, get_relaxed_face_detection_config  # type: ignore
from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService  # type: ignore
from src.ai_services.face_analysis.face_analysis_service import FaceAnalysisService  # type: ignore
from src.api.face_detection_api import router as face_detection_router, init_face_detection_api  # type: ignore
from src.api.face_recognition_api import router as face_recognition_router  # type: ignore
from src.api.face_analysis_api import router as face_analysis_router  # type: ignore

import uvicorn  # type: ignore
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel  # type: ignore

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
    # These are default paths, actual paths will be taken from relaxed_config
    yolov9c_model_path: str = "model/face-detection/yolov9c-face-lindevs.onnx"
    yolov9e_model_path: str = "model/face-detection/yolov9e-face-lindevs.onnx"
    yolov11m_model_path: str = "model/face-detection/yolov11m-face.pt"
    
    # Values below are now superseded by get_relaxed_face_detection_config()
    # max_usable_faces_yolov9: int = 12 # Guide
    # min_agreement_ratio: float = 0.5 # Guide
    # min_quality_threshold: int = 40 # Guide
    # conf_threshold: float = 0.10 # Guide
    # iou_threshold: float = 0.4 # Default, relaxed is 0.35
    # img_size: int = 640


# สร้างแอปพลิเคชัน FastAPI
app = FastAPI(
    title="Face Detection Service API",
    description="บริการตรวจจับใบหน้าที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m (Relaxed Configuration)",
    version="1.0.1" # Version updated for relaxed config
)

# เพิ่ม CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตัวแปรสำหรับเก็บบริการ AI
face_detection_service = None
face_recognition_service = None
face_analysis_service = None
vram_manager = None


@app.on_event("startup")
async def startup_event():
    """
    ฟังก์ชันที่ทำงานเมื่อเริ่มต้นแอปพลิเคชัน
    """
    global face_detection_service, face_recognition_service, face_analysis_service, vram_manager
    
    try:
        # โหลดการตั้งค่า (Settings class is now more for app_name, paths are illustrative)
        settings = Settings()
        
        # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/face-detection", exist_ok=True)
        
        # ตั้งค่า VRAM Manager
        vram_manager_config = {
            "reserved_vram_mb": 512,
            "model_vram_estimates": {
                # Face Detection Models
                "yolov9c-face": 512 * 1024 * 1024,
                "yolov9e-face": 2048 * 1024 * 1024, 
                "yolov11m-face": 2 * 1024 * 1024 * 1024,
                
                # Face Recognition Models  
                "adaface": 89 * 1024 * 1024,
                "arcface": 249 * 1024 * 1024,
                "facenet": 249 * 1024 * 1024,
            }
        }
        
        vram_manager = VRAMManager(vram_manager_config)
        
        # MODIFIED: Get relaxed configuration for FaceDetectionService
        logger.info("Loading RELAXED face detection configuration...")
        face_detection_config = get_relaxed_face_detection_config()

        # Log the specific relaxed values being used from the guide
        logger.info(f"Relaxed - conf_threshold: {face_detection_config.get('conf_threshold')}")
        logger.info(f"Relaxed - max_usable_faces_yolov9: {face_detection_config.get('max_usable_faces_yolov9')}")
        logger.info(f"Relaxed - min_agreement_ratio: {face_detection_config.get('min_agreement_ratio')}")
        logger.info(f"Relaxed - min_quality_threshold: {face_detection_config.get('min_quality_threshold')}")
        logger.info(f"Relaxed - filter_min_quality_final: {face_detection_config.get('filter_min_quality_final')}")

        face_detection_service = FaceDetectionService(vram_manager, face_detection_config)
        
        # โหลดโมเดล Face Detection
        init_success = await face_detection_service.initialize()
        
        if not init_success:
            logger.error("ไม่สามารถโหลดโมเดลตรวจจับใบหน้าได้")
        
        # ตั้งค่าบริการ Face Recognition (config remains the same)
        face_recognition_config = {
            "model_path": "model/face-recognition"
        }
        
        face_recognition_service = FaceRecognitionService(vram_manager, face_recognition_config)
        
        # โหลดโมเดล Face Recognition
        recognition_init = await face_recognition_service.initialize()
        
        if not recognition_init:
            logger.error("ไม่สามารถโหลดโมเดล Face Recognition ได้")
        
        # ตั้งค่าบริการ Face Analysis (Integration Service)
        # Pass the relaxed detection config to analysis service as well
        face_analysis_config = {
            "detection": face_detection_config, # Use the loaded relaxed config
            "recognition": face_recognition_config
        }
        
        face_analysis_service = FaceAnalysisService(vram_manager, face_analysis_config)
        
        # โหลดโมเดล Face Analysis
        analysis_init = await face_analysis_service.initialize()
        
        if not analysis_init:
            logger.error("ไม่สามารถโหลดบริการ Face Analysis ได้")
        
        # ตั้งค่า API
        init_face_detection_api(face_detection_service)
        
        # Initialize Face Recognition API
        from src.api import face_recognition_api
        face_recognition_api.face_recognition_service = face_recognition_service
        face_recognition_api.vram_manager = vram_manager
        
        # Initialize Face Analysis API
        from src.api import face_analysis_api
        face_analysis_api.face_analysis_service = face_analysis_service
        face_analysis_api.vram_manager = vram_manager
        
        logger.info(f"เริ่มต้น {settings.app_name} เรียบร้อยแล้ว (Relaxed Configuration)")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการเริ่มต้นแอปพลิเคชัน: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """
    ฟังก์ชันที่ทำงานเมื่อปิดแอปพลิเคชัน
    """
    global face_detection_service, face_recognition_service, face_analysis_service
    
    try:
        if face_analysis_service:
            await face_analysis_service.cleanup()
        
        if face_recognition_service:
            await face_recognition_service.cleanup()
            
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
app.include_router(face_recognition_router)
app.include_router(face_analysis_router)

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
        "configuration_mode": "Relaxed", # Added status
        "documentation": "/docs",
        "models": ["YOLOv9c", "YOLOv9e", "YOLOv11m"]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
