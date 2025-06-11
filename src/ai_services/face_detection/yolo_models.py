# cSpell:disable
# mypy: ignore-errors
"""
ระบบโมเดล YOLO สำหรับการตรวจจับใบหน้า รองรับ YOLOv9c, YOLOv9e และ YOLOv11m
"""
import time
import os
import logging
import numpy as np
import cv2
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import onnxruntime as ort

logger = logging.getLogger(__name__)

class FaceDetector(ABC):
    """คลาสพื้นฐานสำหรับการตรวจจับใบหน้า"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
    
    @abstractmethod
    def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล"""
        pass
    
    @abstractmethod
    def detect(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพ"""
        pass
    
    @abstractmethod
    def get_input_size(self) -> Tuple[int, int]:
        """ขนาด input ของโมเดล"""
        pass
    
    def preprocess_image(self, image_input) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        แปลงรูปภาพให้เหมาะสมกับการใช้งานกับโมเดล
        รองรับทั้งชื่อไฟล์และ numpy array
        """
        # รองรับทั้งชื่อไฟล์และ numpy array
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพ: {image_input}")
            image = cv2.imread(image_input)
        else:
            image = image_input
            
        if image is None:
            raise ValueError("ไม่สามารถอ่านรูปภาพได้")
            
        # เก็บรูปร่างต้นฉบับ
        original_height, original_width = image.shape[:2]
        
        # ปรับขนาดรูปภาพตามขนาด input ของโมเดล
        target_height, target_width = self.get_input_size()
        scale = min(target_width / original_width, target_height / original_height)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # เพิ่ม padding ให้ได้ขนาดตามที่ต้องการ
        padded_image = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        padded_image[y_offset:y_offset + new_height, 
                    x_offset:x_offset + new_width] = resized_image
        
        # แปลงเป็นรูปแบบที่เหมาะสม
        input_tensor = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        scale_factors = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'original_width': original_width,
            'original_height': original_height
        }
        
        return input_tensor, scale_factors


class YOLOv9ONNXDetector(FaceDetector):
    """
    คลาสสำหรับโมเดล YOLO v9 แบบ ONNX (รองรับทั้ง YOLOv9c และ YOLOv9e)
    """
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)
        self.session = None
        self.input_size = (640, 640)
        self.input_name = None
        self.output_names = None
        self.device = None
        self.model_loaded = False
    
    def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล ONNX"""
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # เพิ่มการตั้งค่าเพื่อลดการใช้หน่วยความจำ
            session_options.enable_mem_pattern = False
            session_options.enable_cpu_mem_arena = False
            
            if device == "cuda" and torch.cuda.is_available():
                try:
                    # ตรวจสอบ VRAM ที่มี
                    available_memory = torch.cuda.get_device_properties(0).total_memory
                    allocated_memory = torch.cuda.memory_allocated(0)
                    free_memory = available_memory - allocated_memory
                    
                    # กำหนดขีดจำกัด VRAM ตามชื่อโมเดล
                    if "yolov9e" in self.model_name.lower():
                        # YOLOv9e ต้องการหน่วยความจำมากกว่า
                        memory_limit = min(2048 * 1024 * 1024, int(free_memory * 0.8))  # 2GB หรือ 80% ของหน่วยความจำที่ว่าง
                    else:
                        # YOLOv9c และโมเดลอื่นๆ
                        memory_limit = min(1024 * 1024 * 1024, int(free_memory * 0.6))  # 1GB หรือ 60% ของหน่วยความจำที่ว่าง
                    
                    logger.info(f"กำลังจัดสรร {memory_limit/1024/1024:.1f}MB VRAM สำหรับ {self.model_name}")
                    
                    # ตั้งค่า CUDA provider สำหรับโมเดลขนาดใหญ่
                    cuda_options = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kSameAsRequested',  # เปลี่ยนจาก kNextPowerOfTwo
                        'gpu_mem_limit': memory_limit,
                        'cudnn_conv_algo_search': 'HEURISTIC',  # เปลี่ยนจาก EXHAUSTIVE เพื่อลดการใช้หน่วยความจำ
                        'do_copy_in_default_stream': True,
                        'enable_cuda_graph': False,  # ปิดเพื่อลดการใช้หน่วยความจำ
                    }
                    
                    providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
                    logger.info(f"โหลดโมเดล {self.model_name} บน GPU พร้อม fallback CPU")
                    
                except Exception as cuda_error:
                    logger.warning(f"ไม่สามารถตั้งค่า CUDA สำหรับ {self.model_name}: {cuda_error}")
                    providers = ['CPUExecutionProvider']
                    device = "cpu"
                    logger.info(f"โหลดโมเดล {self.model_name} บน CPU แทน")
            else:
                providers = ['CPUExecutionProvider']
                device = "cpu"
                logger.info(f"โหลดโมเดล {self.model_name} บน CPU")
            
            # พยายามโหลดโมเดล
            try:
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=session_options,
                    providers=providers
                )
            except Exception as onnx_error:
                # ถ้าโหลดด้วย GPU ไม่สำเร็จ ให้ลองใช้ CPU
                if device == "cuda":
                    logger.warning(f"โหลดโมเดล {self.model_name} บน GPU ไม่สำเร็จ: {onnx_error}")
                    logger.info(f"กำลังลองโหลด {self.model_name} บน CPU...")
                    
                    # ตั้งค่าใหม่สำหรับ CPU
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session_options.enable_mem_pattern = False
                    session_options.enable_cpu_mem_arena = False
                    
                    self.session = ort.InferenceSession(
                        self.model_path,
                        sess_options=session_options,
                        providers=['CPUExecutionProvider']
                    )
                    device = "cpu"
                else:
                    raise onnx_error
            
            # เก็บข้อมูลของโมเดล
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.model_loaded = True
            self.device = device
            
            logger.info(f"โหลดโมเดล {self.model_name} สำเร็จ บนอุปกรณ์: {device}")
            return True
            
        except Exception as e:
            logger.error(f"ไม่สามารถโหลดโมเดล {self.model_name}: {e}")
            return False
    
    def get_input_size(self) -> Tuple[int, int]:
        """ขนาด input ของโมเดล"""
        return self.input_size
        
    def detect(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพด้วย YOLO v9"""
        if not self.model_loaded:
            raise RuntimeError(f"โมเดล {self.model_name} ยังไม่ได้โหลด")
        
        try:
            # แปลงรูปภาพ
            input_tensor, scale_factors = self.preprocess_image(image)
            
            # ล้างหน่วยความจำ GPU ก่อนรันโมเดล (ถ้าใช้ CUDA)
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # รันโมเดล
            start_time = time.time()
            try:
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            except Exception as inference_error:
                # ถ้ารันไม่สำเร็จเนื่องจากหน่วยความจำ ให้ลองล้างแคชและรันอีกครั้ง
                if "memory" in str(inference_error).lower() or "allocation" in str(inference_error).lower():
                    logger.warning(f"เกิดปัญหาหน่วยความจำในการรัน {self.model_name}, กำลังล้างแคชและลองใหม่...")
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # ลองรันอีกครั้ง
                    outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
                else:
                    raise inference_error
                    
            inference_time = time.time() - start_time
            
            # แปลงผลลัพธ์
            detections = self._postprocess_outputs(
                outputs, scale_factors, conf_threshold, iou_threshold
            )
            
            # ล้างหน่วยความจำหลังการใช้งาน
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.debug(f"{self.model_name} ตรวจพบ {len(detections)} ใบหน้า ใช้เวลา {inference_time:.4f} วินาที")
            return detections
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตรวจจับด้วย {self.model_name}: {e}")
            # ล้างหน่วยความจำในกรณีเกิดข้อผิดพลาด
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return []
    
    def detect_faces_raw(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """
        ตรวจจับใบหน้าและส่งคืน raw detection results
        เป็น wrapper สำหรับ detect method
        """
        return self.detect(image, conf_threshold, iou_threshold)

    def has_attribute_detect_faces_raw(self) -> bool:
        """ตรวจสอบว่ามี method detect_faces_raw หรือไม่"""
        return hasattr(self, 'detect_faces_raw')

    def _postprocess_outputs(self, 
                           outputs: List[np.ndarray], 
                           scale_factors: Dict[str, Any],
                           conf_threshold: float,
                           iou_threshold: float) -> List[np.ndarray]:
        """แปลงผลลัพธ์จากโมเดล YOLO v9 - FIXED VERSION"""
        predictions = outputs[0]
        detections = []

        # แก้ไข broadcasting error
        try:
            # ตรวจสอบ shape ของ predictions
            if len(predictions.shape) == 3:
                # Shape: [batch, num_detections, features]
                batch_predictions = predictions[0]  # เอาแค่ batch แรก
            elif len(predictions.shape) == 2:
                # Shape: [num_detections, features]
                batch_predictions = predictions
            else:
                logger.warning(f"Unexpected predictions shape: {predictions.shape}")
                return []

            for pred in batch_predictions:
                try:
                    # ตรวจสอบความยาวของ prediction
                    if len(pred) < 5:
                        continue

                    # แก้ไข broadcasting error - ใช้ indexing แบบชัดเจน
                    x_center = float(pred[0])
                    y_center = float(pred[1])
                    width = float(pred[2])
                    height = float(pred[3])
                    confidence = float(pred[4])

                    if confidence < conf_threshold:
                        continue

                    # แปลงจาก center format เป็น corner format
                    x1 = x_center - (width / 2)
                    y1 = y_center - (height / 2)
                    x2 = x_center + (width / 2)
                    y2 = y_center + (height / 2)

                    # ปรับขนาดตาม input size
                    input_size = self.get_input_size()
                    x1 = x1 * input_size[1]
                    y1 = y1 * input_size[0]
                    x2 = x2 * input_size[1]
                    y2 = y2 * input_size[0]

                    # แปลงกลับเป็นพิกัดต้นฉบับ
                    scale = scale_factors['scale']
                    x_offset = scale_factors['x_offset']
                    y_offset = scale_factors['y_offset']

                    x1 = (x1 - x_offset) / scale
                    y1 = (y1 - y_offset) / scale
                    x2 = (x2 - x_offset) / scale
                    y2 = (y2 - y_offset) / scale

                    # Clip to image bounds
                    x1 = max(0, min(x1, scale_factors['original_width']))
                    y1 = max(0, min(y1, scale_factors['original_height']))
                    x2 = max(0, min(x2, scale_factors['original_width']))
                    y2 = max(0, min(y2, scale_factors['original_height']))

                    # สร้าง detection array [x1, y1, x2, y2, confidence]
                    detection = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)
                    detections.append(detection)

                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"Error processing prediction: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return []

        # Apply NMS
        if detections:
            detections_array = np.array(detections)
            final_detections = self._nms(detections_array, iou_threshold)
            return final_detections
        else:
            return []

    def _nms(self, detections: np.ndarray, iou_threshold: float) -> List[np.ndarray]:
        """
        Perform Non-Maximum Suppression (NMS) on detections.
        Args:
            detections: Array of detections, where each row is [x1, y1, x2, y2, confidence, (optional class_id)].
            iou_threshold: IoU threshold for suppression.
        Returns:
            List of filtered detections.
        """
        if detections.shape[0] == 0:
            return []

        # Extract coordinates and scores
        # Assuming detections are [x1, y1, x2, y2, confidence, ...]
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]  # Sort by confidence in descending order

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(detections[i])
            
            if order.size == 1: # No more boxes to compare
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1] # +1 because order[0] was the current box

        return keep


class YOLOv11Detector(FaceDetector):
    """
    คลาสสำหรับโมเดล YOLO v11 (Ultralytics) สำหรับการตรวจจับใบหน้า
    """
    
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)
        self.model = None
        self.input_size = (640, 640)
        self.device = None
        self.model_loaded = False
    
    def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล YOLOv11"""
        try:
            # เช็ค CUDA
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA ไม่พร้อมใช้งาน จะใช้ CPU แทน")
                device = "cpu"
            
            # โหลดโมเดลด้วย Ultralytics
            from ultralytics import YOLO
            
            self.model = YOLO(self.model_path)
            self.device = device
            self.model_loaded = True
            
            logger.info(f"โหลดโมเดล {self.model_name} สำเร็จ บนอุปกรณ์: {device}")
            return True
            
        except Exception as e:
            logger.error(f"ไม่สามารถโหลดโมเดล {self.model_name}: {e}")
            return False
    
    def get_input_size(self) -> Tuple[int, int]:
        """ขนาด input ของโมเดล"""
        return self.input_size
    
    def detect(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพด้วย YOLO v11"""
        if not self.model_loaded:
            raise RuntimeError(f"โมเดล {self.model_name} ยังไม่ได้โหลด")
        
        try:
            # รองรับทั้งชื่อไฟล์และ numpy array
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพ: {image}")
                # ใช้ path โดยตรงสำหรับ YOLO
                img_input = image
            else:
                # ต้องแปลงจาก numpy array เป็นรูปภาพเพื่อใช้กับ YOLO
                if not isinstance(image, np.ndarray):
                    raise ValueError("รูปภาพต้องเป็น numpy array")
                
                # สร้างไฟล์ชั่วคราว
                temp_img_path = "temp_yolov11_input.jpg"
                cv2.imwrite(temp_img_path, image)
                img_input = temp_img_path
            
            # รันโมเดล
            start_time = time.time()
            results = self.model(img_input, conf=conf_threshold, iou=iou_threshold, device=self.device)
            inference_time = time.time() - start_time
            
            # ถ้าใช้ไฟล์ชั่วคราว ให้ลบทิ้ง
            if isinstance(image, np.ndarray) and os.path.exists("temp_yolov11_input.jpg"):
                os.remove("temp_yolov11_input.jpg")
            
            # แปลงผลลัพธ์เป็นรูปแบบเดียวกับ YOLOv9
            detections = self._convert_results(results)
            
            logger.debug(f"{self.model_name} ตรวจพบ {len(detections)} ใบหน้า ใช้เวลา {inference_time:.4f} วินาที")
            return detections
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตรวจจับด้วย {self.model_name}: {e}")
            # ถ้าใช้ไฟล์ชั่วคราว ให้ลบทิ้ง
            if isinstance(image, np.ndarray) and os.path.exists("temp_yolov11_input.jpg"):
                os.remove("temp_yolov11_input.jpg")
            return []
    
    def detect_faces_raw(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """
        ตรวจจับใบหน้าและส่งคืน raw detection results
        เป็น wrapper สำหรับ detect method
        """
        return self.detect(image, conf_threshold, iou_threshold)

    def has_attribute_detect_faces_raw(self) -> bool:
        """ตรวจสอบว่ามี method detect_faces_raw หรือไม่"""
        return hasattr(self, 'detect_faces_raw')
    
    def _convert_results(self, results) -> List[np.ndarray]:
        """แปลงผลลัพธ์จาก YOLO v11 ให้อยู่ในรูปแบบเดียวกับ YOLOv9"""
        detections = []
        
        # รับผลลัพธ์จากโมเดล Ultralytics
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # รับค่าพิกัดและ confidence
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # สร้าง detection array [x1, y1, x2, y2, confidence]
                detection = np.array([x1, y1, x2, y2, confidence])
                detections.append(detection)
        
        return detections
