# cSpell:disable
"""
บริการตรวจจับใบหน้า (Face Detection Service) ที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m
"""
import time
import logging
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional, Union

from .yolo_models import YOLOv9ONNXDetector, YOLOv11Detector
from .utils import BoundingBox, FaceDetection, DetectionResult, calculate_face_quality

# ไม่ import VRAMManager แต่สร้าง stub class ขึ้นมาแทน
class VRAMManager:
    """Stub class for VRAMManager."""
    async def request_model_allocation(self, *args, **kwargs):
        class Allocation:
            class Location:
                value = "cpu"
            location = Location()
        return Allocation()
    
    async def release_model_allocation(self, *args, **kwargs):
        return True
        
    async def get_vram_status(self, *args, **kwargs):
        return {"status": "stub", "available": 0}

logger = logging.getLogger(__name__)

class FaceDetectionService:
    """
    บริการตรวจจับใบหน้าอัจฉริยะที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m
    """
    def __init__(self, vram_manager: VRAMManager, config: Dict[str, Any]):
        """
        ตั้งค่าเริ่มต้นสำหรับบริการตรวจจับใบหน้า
        
        Args:
            vram_manager: ตัวจัดการหน่วยความจำ GPU
            config: การตั้งค่าสำหรับบริการ
        """
        self.vram_manager = vram_manager
        self.config = config
        self.models: dict[str, Union[YOLOv9ONNXDetector, YOLOv11Detector]] = {}
        self.model_stats: dict[str, dict[str, Union[float, int]]] = {}
        
        # เกณฑ์การตัดสินใจเลือกโมเดล
        self.decision_criteria = {
            'max_usable_faces_yolov9': int(config.get('max_usable_faces_yolov9', 8)),
            'min_agreement_ratio': float(config.get('min_agreement_ratio', 0.7)),
            'min_quality_threshold': int(config.get('min_quality_threshold', 60)),
            'iou_threshold': float(config.get('iou_threshold', 0.5))
        }
        
        # พารามิเตอร์การตรวจจับ
        self.detection_params = {
            'conf_threshold': config.get('conf_threshold', 0.15),
            'iou_threshold': config.get('iou_threshold', 0.4),
            'img_size': config.get('img_size', 640)
        }
        
        self.yolov9c_model_path = config.get('yolov9c_model_path', 'model/face-detection/yolov9c-face-lindevs.onnx')
        self.yolov9e_model_path = config.get('yolov9e_model_path', 'model/face-detection/yolov9e-face-lindevs.onnx')
        self.yolov11m_model_path = config.get('yolov11m_model_path', 'model/face-detection/yolov11m-face.pt')
        
        self.models_loaded = False
    
    async def initialize(self) -> bool:
        """
        โหลดโมเดลตรวจจับใบหน้าทั้งหมด
        
        Returns:
            สถานะการโหลดโมเดล
        """
        try:
            logger.info("กำลังโหลดโมเดลตรวจจับใบหน้าาทั้งหมด...")
            
            # ขอจัดสรร VRAM สำหรับโมเดล YOLOv9c
            yolov9c_allocation = await self.vram_manager.request_model_allocation(
                "yolov9c-face", "high", "face_detection_service"
            )
            
            # โหลดโมเดล YOLOv9c
            self.models['yolov9c'] = YOLOv9ONNXDetector(self.yolov9c_model_path, "YOLOv9c")
            yolov9c_device = "cuda" if yolov9c_allocation.location.value == "gpu" else "cpu"
            self.models['yolov9c'].load_model(yolov9c_device)
            
            # ขอจัดสรร VRAM สำหรับโมเดล YOLOv9e
            yolov9e_allocation = await self.vram_manager.request_model_allocation(
                "yolov9e-face", "high", "face_detection_service"
            )
            
            # โหลดโมเดล YOLOv9e
            self.models['yolov9e'] = YOLOv9ONNXDetector(self.yolov9e_model_path, "YOLOv9e")
            yolov9e_device = "cuda" if yolov9e_allocation.location.value == "gpu" else "cpu"
            self.models['yolov9e'].load_model(yolov9e_device)
            
            # ขอจัดสรร VRAM สำหรับโมเดล YOLOv11m
            yolov11m_allocation = await self.vram_manager.request_model_allocation(
                "yolov11m-face", "critical", "face_detection_service"
            )
            
            # โหลดโมเดล YOLOv11m
            self.models['yolov11m'] = YOLOv11Detector(self.yolov11m_model_path, "YOLOv11m")
            yolov11m_device = "cuda" if yolov11m_allocation.location.value == "gpu" else "cpu"
            self.models['yolov11m'].load_model(yolov11m_device)
            
            self.models_loaded = True
            logger.info("โหลดโมเดลตรวจจับใบหน้าเรียบร้อยแล้ว")
            return True
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            return False
    
    async def detect_faces(self, 
                         image_input: Union[str, np.ndarray],
                         model_name: Optional[str] = None,
                         conf_threshold: Optional[float] = None,
                         iou_threshold: Optional[float] = None) -> DetectionResult:
        """
        ตรวจจับใบหน้าในรูปภาพโดยเลือกโมเดลที่เหมาะสมที่สุดโดยอัตโนมัติ
        
        Args:
            image_input: ชื่อไฟล์รูปภาพหรือ numpy array
            model_name: ชื่อโมเดลที่ต้องการใช้ ('yolov9c', 'yolov9e', 'yolov11m' หรือ 'auto')
            conf_threshold: ระดับความมั่นใจขั้นต่ำ
            iou_threshold: ค่า IoU threshold สำหรับ NMS
        
        Returns:
            ผลลัพธ์การตรวจจับใบหน้า
        """
        if not self.models_loaded:
            raise RuntimeError("ยังไม่ได้โหลดโมเดล โปรดเรียก initialize() ก่อน")
        
        # ตั้งค่าพารามิเตอร์
        conf_threshold = conf_threshold or self.detection_params['conf_threshold']
        iou_threshold = iou_threshold or self.detection_params['iou_threshold']
        
        start_time = time.time()
        
        # โหลดรูปภาพ
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพ: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"ไม่สามารถอ่านรูปภาพได้: {image_input}")
        else:
            image = image_input
        
        # ถ้าระบุโมเดลมา ให้ใช้โมเดลนั้น
        if model_name in ['yolov9c', 'yolov9e', 'yolov11m']:
            detections = self._detect_with_model(
                image, model_name, conf_threshold, iou_threshold
            )
            return self._create_result(detections, image.shape, time.time() - start_time, model_name)
        
        # ตรวจจับด้วยระบบอัจฉริยะที่เลือกโมเดลอัตโนมัติ
        return await self._intelligent_detect(image, conf_threshold, iou_threshold, start_time)
    
    def _detect_with_model(self, 
                         image: np.ndarray, 
                         model_name: str,
                         conf_threshold: float,
                         iou_threshold: float) -> List[FaceDetection]:
        """
        ตรวจจับใบหน้าด้วยโมเดลที่ระบุ
        
        Args:
            image: รูปภาพ (numpy array)
            model_name: ชื่อโมเดล ('yolov9c', 'yolov9e', 'yolov11m')
            conf_threshold: ระดับความมั่นใจขั้นต่ำ
            iou_threshold: ค่า IoU threshold สำหรับ NMS
        
        Returns:
            รายการใบหน้าที่ตรวจพบ
        """
        # บันทึกเวลาเริ่มต้น
        model_start_time = time.time()
        
        # ตรวจจับใบหน้า
        detections_raw = self.models[model_name].detect(
            image, conf_threshold, iou_threshold
        )
        
        # บันทึกเวลาที่ใช้
        inference_time = time.time() - model_start_time
        
        # แปลงผลลัพธ์
        face_detections = []
        for det in detections_raw:
            bbox = BoundingBox.from_array(det)
            # แปลง image.shape[:2] เป็น tuple[int, int] เพื่อให้ตรงกับ signature ของฟังก์ชัน
            image_size = (int(image.shape[0]), int(image.shape[1]))
            quality_score = calculate_face_quality(bbox, image_size)
            
            face = FaceDetection(
                bbox=bbox,
                quality_score=quality_score,
                model_used=model_name,
                processing_time=inference_time
            )
            face_detections.append(face)
        
        # บันทึกสถิติ
        quality_scores = [f.quality_score for f in face_detections if f.quality_score is not None]
        self.model_stats[model_name] = {
            'last_inference_time': inference_time,
            'face_count': len(face_detections),
            'avg_quality': float(np.mean(quality_scores)) if quality_scores else 0.0
        }
        
        return face_detections
    
    async def _intelligent_detect(self,
                               image: np.ndarray,
                               conf_threshold: float,
                               iou_threshold: float,
                               start_time: float) -> DetectionResult:
        """
        ระบบตรวจจับอัจฉริยะที่เลือกโมเดลอัตโนมัติ
        
        Args:
            image: รูปภาพ (numpy array)
            conf_threshold: ระดับความมั่นใจขั้นต่ำ
            iou_threshold: ค่า IoU threshold สำหรับ NMS
            start_time: เวลาเริ่มต้น
            
        Returns:
            ผลลัพธ์การตรวจจับใบหน้า
        """
        logger.debug("กำลังใช้ระบบตรวจจับอัจฉริยะ...")
        
        # ขั้นตอน 1: ทดสอบด้วย YOLOv9c ซึ่งเร็วที่สุด
        yolov9c_detections = self._detect_with_model(
            image, 'yolov9c', conf_threshold, iou_threshold
        )
        
        # ถ้าไม่พบใบหน้าเลย ลองใช้ YOLOv11m ซึ่งแม่นยกว่า
        if not yolov9c_detections:
            logger.debug("YOLOv9c ไม่พบใบหน้า กำลังลองใช้ YOLOv11m...")
            yolov11m_detections = self._detect_with_model(
                image, 'yolov11m', conf_threshold, iou_threshold
            )
            return self._create_result(
                yolov11m_detections, image.shape, time.time() - start_time, 'yolov11m'
            )
        
        # ถ้าพบใบหน้าน้อยกว่าหรือเท่ากับค่าที่กำหนด ให้ใช้ YOLOv9c เลย
        max_faces = self.decision_criteria['max_usable_faces_yolov9']
        if len(yolov9c_detections) <= max_faces:
            logger.debug(f"YOLOv9c พบ {len(yolov9c_detections)} ใบหน้า (≤{max_faces}) ใช้ผลลัพธ์นี้เลย")
            return self._create_result(
                yolov9c_detections, image.shape, time.time() - start_time, 'yolov9c'
            )
        
        # ถ้าพบใบหน้าเยอะเกินไป ให้ทดสอบด้วย YOLOv9e ต่อ
        logger.debug(f"YOLOv9c พบ {len(yolov9c_detections)} ใบหน้า (>{max_faces}) กำลังทดสอบด้วย YOLOv9e...")
        yolov9e_detections = self._detect_with_model(
            image, 'yolov9e', conf_threshold, iou_threshold
        )
        
        # เปรียบเทียบผลลัพธ์ระหว่าง YOLOv9c และ YOLOv9e
        agreement = self._calculate_agreement(
            yolov9c_detections, yolov9e_detections, self.decision_criteria['iou_threshold']
        )
        
        # ถ้าผลลัพธ์สอดคล้องกัน และมีคุณภาพดี ให้ใช้ YOLOv9e
        min_agreement = self.decision_criteria['min_agreement_ratio']
        if agreement >= min_agreement:
            logger.debug(f"YOLOv9c และ YOLOv9e เห็นด้วยกัน {agreement:.1%} (≥{min_agreement:.1%}) ใช้ YOLOv9e")
            return self._create_result(
                yolov9e_detections, image.shape, time.time() - start_time, 'yolov9e'
            )
        
        # ถ้าผลลัพธ์ไม่สอดคล้องกัน ให้ลองใช้ YOLOv11m ซึ่งแม่นยำที่สุด
        logger.debug(f"YOLOv9c และ YOLOv9e เห็นด้วยกันเพียง {agreement:.1%} (<{min_agreement:.1%}) กำลังใช้ YOLOv11m...")
        yolov11m_detections = self._detect_with_model(
            image, 'yolov11m', conf_threshold, iou_threshold
        )
        
        return self._create_result(
            yolov11m_detections, image.shape, time.time() - start_time, 'yolov11m'
        )
    
    def _calculate_agreement(self, 
                          detections1: List[FaceDetection], 
                          detections2: List[FaceDetection],
                          iou_threshold: float) -> float:
        """
        คำนวณความสอดคล้องระหว่างผลลัพธ์การตรวจจับสองชุด
        
        Args:
            detections1: ผลลัพธ์ชุดที่ 1
            detections2: ผลลัพธ์ชุดที่ 2
            iou_threshold: ค่า IoU threshold สำหรับพิจารณาว่าตรงกัน
            
        Returns:
            สัดส่วนความสอดคล้อง (0.0-1.0)
        """
        if not detections1 or not detections2:
            return 0.0
        
        # จำนวนใบหน้าทั้งหมด
        total_faces = max(len(detections1), len(detections2))
        
        # แปลงเป็น numpy array เพื่อความสะดวก
        boxes1 = np.array([d.bbox.to_array()[:4] for d in detections1])  # x1, y1, x2, y2
        boxes2 = np.array([d.bbox.to_array()[:4] for d in detections2])  # x1, y1, x2, y2
        
        # นับจำนวนใบหน้าที่ตรงกัน
        matched_count = 0
        
        # สำหรับแต่ละกล่องในชุดแรก
        for box1 in boxes1:
            best_iou = 0.0
            
            # หา IoU ที่ดีที่สุดกับกล่องในชุดที่สอง
            for i, box2 in enumerate(boxes2):
                iou = self._calculate_iou(box1, box2)
                if iou > best_iou:
                    best_iou = iou
            
            # ถ้ามี IoU ที่ดีพอ ถือว่าตรงกัน
            if best_iou >= iou_threshold:
                matched_count += 1
        
        # คำนวณสัดส่วน
        return matched_count / total_faces
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        คำนวณค่า IoU (Intersection over Union) ระหว่างสองกล่อง
        
        Args:
            box1: กล่องที่ 1 [x1, y1, x2, y2]
            box2: กล่องที่ 2 [x1, y1, x2, y2]
            
        Returns:
            ค่า IoU (0.0-1.0)
        """
        # พื้นที่ทับซ้อน
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # พื้นที่ของแต่ละกล่อง
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # คำนวณ IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou
    
    def _create_result(self, 
                     detections: List[FaceDetection], 
                     image_shape: Tuple[int, ...],
                     total_time: float,
                     model_used: str) -> DetectionResult:
        """
        สร้างผลลัพธ์การตรวจจับใบหน้าในรูปแบบมาตรฐาน
        
        Args:
            detections: รายการใบหน้าที่ตรวจพบ
            image_shape: ขนาดรูปภาพ (height, width, channels)
            total_time: เวลาที่ใช้ทั้งหมด
            model_used: โมเดลที่ใช้
            
        Returns:
            ผลลัพธ์การตรวจจับใบหน้า
        """
        # แปลง image_shape เป็น Tuple[int, int, int]
        shape = (image_shape[0], image_shape[1], image_shape[2] if len(image_shape) > 2 else 3)
        
        return DetectionResult(
            faces=detections,
            image_shape=shape,
            total_processing_time=total_time,
            model_used=model_used,
            fallback_used=False
        )
    
    async def get_service_info(self) -> Dict[str, Any]:
        """
        ดูข้อมูลของบริการตรวจจับใบหน้า
        
        Returns:
            ข้อมูลบริการและสถิติการใช้งาน
        """
        vram_status = await self.vram_manager.get_vram_status()
        
        return {
            "service_name": "Face Detection Service",
            "models_loaded": self.models_loaded,
            "available_models": list(self.models.keys()),
            "model_stats": self.model_stats,
            "decision_criteria": self.decision_criteria,
            "detection_params": self.detection_params,
            "vram_status": vram_status
        }
    
    async def cleanup(self) -> bool:
        """
        ทำความสะอาดทรัพยากร
        
        Returns:
            สถานะการทำความสะอาด
        """
        try:
            # คืนทรัพยากร VRAM
            await self.vram_manager.release_model_allocation("yolov9c-face")
            await self.vram_manager.release_model_allocation("yolov9e-face")
            await self.vram_manager.release_model_allocation("yolov11m-face")
            
            # ล้างข้อมูลโมเดล
            self.models = {}
            self.models_loaded = False
            
            logger.info("ทำความสะอาดทรัพยากรเรียบร้อยแล้ว")
            return True
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการทำความสะอาดทรัพยากร: {e}")
            return False
