# cSpell:disable
"""
สคริปต์สำหรับทดสอบบริการตรวจจับใบหน้า
"""
import os
import sys
import asyncio
import logging
import time
import cv2
import json

# เพิ่มโฟลเดอร์หลักไปยัง Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_services.common.vram_manager import VRAMManager
from src.ai_services.face_detection.face_detection_service import FaceDetectionService
from src.ai_services.face_detection.utils import save_detection_image

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_face_detection():
    """
    ทดสอบบริการตรวจจับใบหน้า
    """    # ตั้งค่า VRAM Manager
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
        "yolov9c_model_path": "model/face-detection/yolov9c-face-lindevs.onnx",
        "yolov9e_model_path": "model/face-detection/yolov9e-face-lindevs.onnx",
        "yolov11m_model_path": "model/face-detection/yolov11m-face.pt",
        "max_usable_faces_yolov9": 8,
        "min_agreement_ratio": 0.7,
        "min_quality_threshold": 60,
        "conf_threshold": 0.15,
        "iou_threshold": 0.4,
        "img_size": 640
    }
    
    face_detection_service = FaceDetectionService(vram_manager, face_detection_config)
    
    # โหลดโมเดล
    logger.info("กำลังโหลดโมเดล...")
    init_success = await face_detection_service.initialize()
    
    if not init_success:
        logger.error("ไม่สามารถโหลดโมเดลตรวจจับใบหน้าได้")
        return
    
    # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
    output_dir = "output/face-detection"
    os.makedirs(output_dir, exist_ok=True)
    
    # รายการรูปภาพสำหรับทดสอบ
    test_image_dir = "test_images"
    test_images = []
    
    for filename in os.listdir(test_image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(test_image_dir, filename))
    
    # ทดสอบการตรวจจับใบหน้า
    logger.info(f"กำลังทดสอบกับ {len(test_images)} รูปภาพ...")
    
    results_summary = []
    
    for i, image_path in enumerate(test_images):
        logger.info(f"กำลังทดสอบรูปภาพที่ {i+1}/{len(test_images)}: {image_path}")
        
        # อ่านรูปภาพ
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"ไม่สามารถอ่านรูปภาพได้: {image_path}")
            continue
        
        # ตรวจจับใบหน้าด้วยระบบอัจฉริยะ (auto)
        start_time = time.time()
        result = await face_detection_service.detect_faces(image)
        detection_time = time.time() - start_time
        
        # บันทึกผลลัพธ์
        filename = os.path.basename(image_path)
        output_filename = f"result_{filename}"
        output_path = save_detection_image(image, result.faces, output_dir, output_filename)
        
        # เพิ่มสรุปผลลัพธ์
        result_json = result.to_dict()
        result_json["image_path"] = image_path
        result_json["output_path"] = output_path
        results_summary.append(result_json)
        
        logger.info(f"ตรวจพบ {len(result.faces)} ใบหน้า ด้วยโมเดล {result.model_used} ใช้เวลา {detection_time:.4f} วินาที")
    
    # บันทึกสรุปผลลัพธ์
    with open(os.path.join(output_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    # สรุปผลการทดสอบ
    total_faces = sum(len(r["faces"]) for r in results_summary)
    total_images = len(results_summary)
    total_time = sum(r["total_processing_time"] for r in results_summary)
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    
    # สรุปตามโมเดลที่ใช้
    model_stats = {}
    for r in results_summary:
        model = r["model_used"]
        if model not in model_stats:
            model_stats[model] = {"count": 0, "faces": 0, "time": 0}
        
        model_stats[model]["count"] += 1
        model_stats[model]["faces"] += len(r["faces"])
        model_stats[model]["time"] += r["total_processing_time"]
    
    for model, stats in model_stats.items():
        avg_time = stats["time"] / stats["count"] if stats["count"] > 0 else 0
        avg_faces = stats["faces"] / stats["count"] if stats["count"] > 0 else 0
        model_stats[model]["avg_time"] = avg_time
        model_stats[model]["avg_faces"] = avg_faces
    
    # พิมพ์สรุปผล
    logger.info("===== สรุปผลการทดสอบ =====")
    logger.info(f"รูปภาพทั้งหมด: {total_images}")
    logger.info(f"ใบหน้าที่ตรวจพบทั้งหมด: {total_faces}")
    logger.info(f"เวลาเฉลี่ยต่อรูป: {avg_time_per_image:.4f} วินาที")
    logger.info("สถิติตามโมเดล:")
    
    for model, stats in model_stats.items():
        logger.info(f"  {model}: ใช้ {stats['count']} ครั้ง, พบ {stats['faces']} ใบหน้า, เวลาเฉลี่ย {stats['avg_time']:.4f} วินาที")
    
    # ทำความสะอาด
    logger.info("กำลังทำความสะอาดทรัพยากร...")
    await face_detection_service.cleanup()
    
    logger.info("ทดสอบเสร็จสิ้น ผลลัพธ์ถูกบันทึกไว้ที่ " + output_dir)


if __name__ == "__main__":
    asyncio.run(test_face_detection())
