# cSpell:disable
# mypy: ignore-errors
"""
Face Recognition Model Selector
ระบบเลอกโมเดลอจฉรยะตาม requirement และทรพยากร
"""

from typing import Dict, Any
import logging

from .models import RecognitionModel
from ..common.vram_manager import VRAMManager


class FaceRecognitionModelSelector:
    """
    ระบบเลอกโมเดล Face Recognition อตโนมต
    เลอกโมเดลทเหมาะสมตามทรพยากรทมและความตองการดานประสทธภาพ
    """
    
    def __init__(self, vram_manager: VRAMManager, models_info: Dict[str, Dict[str, Any]]):
        """
        สรางระบบเลอกโมเดลอตโนมต
        
        Args:
            vram_manager: ระบบจดการ VRAM
            models_info: ขอมลของโมเดลแตละตว ในรปแบบ
                         {model_name: {"size_bytes": int, "accuracy": float, "speed": float}}
                         - accuracy และ speed เปนคาระหวาง 0-1 (สงแปลวาด)
        """
        self.vram_manager = vram_manager
        self.models_info = models_info
        self.logger = logging.getLogger(__name__)
    
    async def select_model(self, preference: str = "balanced") -> RecognitionModel:
        """
        เลือกโมเดลที่เหมาะสมตามความต้องการและทรัพยากรที่มี
        
        Args:
            preference: ความต้องการ
                - "speed": เน้นความเร็ว
                - "accuracy": เน้นความแม่นยำ
                - "balanced": สมดุลระหว่างความเร็วและความแม่นยำ
                - "minimal_memory": ใช้หน่วยความจำน้อยที่สุด
        
        Returns:
            RecognitionModel: โมเดลที่เหมาะสม
        """
        # เช็คหน่วยความจำที่มีอยู่
        available_vram = await self.vram_manager.get_available_memory()
        
        # กรองโมเดลทเหมาะกบหนวยความจำทม
        valid_models = {}
        for model_name, model_info in self.models_info.items():
            model_size = model_info.get("size_bytes", 0)
            if model_size <= available_vram:
                valid_models[model_name] = model_info
        
        # ถาไมมโมเดลทเหมาะสม ใหเลอกโมเดลทเลกทสด
        if not valid_models:
            self.logger.warning("ไมมโมเดลทเหมาะกบ VRAM ทมอย กำลงเลอกโมเดลทเลกทสด")
            smallest_model = min(self.models_info.items(), key=lambda x: x[1].get("size_bytes", float("inf")))
            return RecognitionModel[smallest_model[0]]
        
        # เลอกโมเดลตามความตองการ
        if preference == "speed":
            # เลอกโมเดลทเรวทสด
            selected = max(valid_models.items(), key=lambda x: x[1].get("speed", 0))
        elif preference == "accuracy":
            # เลอกโมเดลทแมนยำทสด
            selected = max(valid_models.items(), key=lambda x: x[1].get("accuracy", 0))
        elif preference == "minimal_memory":
            # เลอกโมเดลทใชหนวยความจำนอยทสด
            selected = min(valid_models.items(), key=lambda x: x[1].get("size_bytes", float("inf")))
        else:  # balanced (default)
            # คำนวณคะแนนรวม (40% speed, 40% accuracy, 20% memory efficiency)
            best_score = -1
            selected = None
            
            for model_name, model_info in valid_models.items():
                speed_score = model_info.get("speed", 0) * 0.4
                accuracy_score = model_info.get("accuracy", 0) * 0.4
                
                # หนวยความจำ: ยงใชนอยยงด
                max_size = max(m.get("size_bytes", 0) for m in self.models_info.values())
                memory_score = (1 - model_info.get("size_bytes", 0) / max_size) * 0.2
                
                total_score = speed_score + accuracy_score + memory_score
                
                if total_score > best_score:
                    best_score = total_score
                    selected = (model_name, model_info)
        
        if selected is None:
            # Fallback ถาไมสามารถเลอกได
            self.logger.warning("ไมสามารถเลอกโมเดลตามเงอนไขได กำลงใชคาเรมตน")
            return RecognitionModel.ADAFACE
        
        model_name = selected[0]
        self.logger.info(f"เลอกโมเดล {model_name} ตามความตองการ: {preference}")
        
        return RecognitionModel[model_name]
