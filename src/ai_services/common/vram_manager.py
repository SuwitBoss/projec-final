# cSpell:disable
"""
VRAM Manager สำหรับจัดการหน่วยความจำ GPU ในระบบ AI
"""
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)

class AllocationPriority(Enum):
    CRITICAL = "critical"  # ต้องอยู่บน GPU เสมอ
    HIGH = "high"          # ควรอยู่บน GPU
    MEDIUM = "medium"      # เป็นตัวเลือก
    LOW = "low"            # GPU ถ้าว่าง

class AllocationLocation(Enum):
    GPU = "gpu"
    CPU = "cpu"

@dataclass
class ModelAllocation:
    model_id: str
    priority: AllocationPriority
    service_id: str
    location: AllocationLocation
    vram_allocated: int
    status: str

class VRAMManager:
    """
    ระบบบริหารจัดการหน่วยความจำ GPU สำหรับโมเดล AI
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_allocations: Dict[str, ModelAllocation] = {}
        self.total_vram = self._get_total_vram()
        self.allocated_vram = 0
        self.lock = asyncio.Lock()
        
    def _get_total_vram(self) -> int:
        """ตรวจสอบขนาด VRAM ทั้งหมดที่มี"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return torch.cuda.get_device_properties(device).total_memory
        return 0
    
    async def request_model_allocation(
        self,
        model_id: str,
        priority: str,
        service_id: str,
        vram_required: Optional[int] = None
    ) -> ModelAllocation:
        """
        ขอจัดสรร VRAM สำหรับโมเดล
        """
        async with self.lock:
            # ตรวจสอบว่ามี GPU หรือไม่
            if self.total_vram == 0:
                logger.warning(f"ไม่มี GPU หรือ VRAM ไม่พอ สำหรับโมเดล {model_id}")
                return ModelAllocation(
                    model_id=model_id,
                    priority=AllocationPriority(priority),
                    service_id=service_id,
                    location=AllocationLocation.CPU,
                    vram_allocated=0,
                    status="fallback_to_cpu"
                )
                
            # ถ้าโมเดลถูกโหลดอยู่แล้ว
            if model_id in self.model_allocations:
                allocation = self.model_allocations[model_id]
                logger.info(f"โมเดล {model_id} ถูกโหลดอยู่แล้วที่ {allocation.location}")
                return allocation
            
            # คำนวณขนาด VRAM ที่ต้องการ
            if vram_required is None:
                # ใช้ค่าประมาณการจากการตั้งค่า
                vram_required = self.config.get("model_vram_estimates", {}).get(model_id, 512 * 1024 * 1024)  # 512MB เป็นค่าเริ่มต้น
            
            # ตรวจสอบว่ามี VRAM พอหรือไม่
            available_vram = self.total_vram - self.allocated_vram
            
            # ถ้ามี VRAM พอ
            if available_vram >= vram_required or priority == AllocationPriority.CRITICAL.value:
                # สำหรับ CRITICAL ถ้า VRAM ไม่พอ จะต้องย้ายโมเดลอื่นออก
                if available_vram < vram_required and priority == AllocationPriority.CRITICAL.value:
                    self._free_vram_for_critical_model(vram_required - available_vram)
                
                # จัดสรร VRAM
                self.allocated_vram += vram_required
                allocation = ModelAllocation(
                    model_id=model_id,
                    priority=AllocationPriority(priority),
                    service_id=service_id,
                    location=AllocationLocation.GPU,
                    vram_allocated=vram_required,
                    status="allocated_on_gpu"
                )
                self.model_allocations[model_id] = allocation
                logger.info(f"จัดสรร {vram_required/1024/1024:.1f}MB VRAM สำหรับโมเดล {model_id}")
                return allocation
            else:
                # ถ้า VRAM ไม่พอ ใช้ CPU แทน
                logger.warning(f"VRAM ไม่พอสำหรับโมเดล {model_id} ({vram_required/1024/1024:.1f}MB > {available_vram/1024/1024:.1f}MB)")
                allocation = ModelAllocation(
                    model_id=model_id,
                    priority=AllocationPriority(priority),
                    service_id=service_id,
                    location=AllocationLocation.CPU,
                    vram_allocated=0,
                    status="fallback_to_cpu_low_vram"
                )
                self.model_allocations[model_id] = allocation
                return allocation
    
    def _free_vram_for_critical_model(self, vram_needed: int) -> None:
        """
        ย้ายโมเดลที่มีความสำคัญน้อยกว่าออกจาก GPU เพื่อให้มี VRAM พอสำหรับโมเดลที่สำคัญกว่า
        """
        # เรียงลำดับโมเดลตามความสำคัญจากน้อยไปมาก
        candidates = sorted(
            [a for a in self.model_allocations.values() if a.location == AllocationLocation.GPU],
            key=lambda x: x.priority.value
        )
        
        vram_freed = 0
        for allocation in candidates:
            # ข้ามโมเดลที่มีความสำคัญสูงสุด
            if allocation.priority == AllocationPriority.CRITICAL:
                continue
                
            # ย้ายโมเดลออกจาก GPU
            vram_freed += allocation.vram_allocated
            logger.info(f"ย้ายโมเดล {allocation.model_id} ออกจาก GPU เพื่อเพิ่มพื้นที่")
            
            # อัปเดตสถานะ
            allocation.location = AllocationLocation.CPU
            allocation.status = "moved_to_cpu_for_critical"
            allocation.vram_allocated = 0
            
            # ลดจำนวน VRAM ที่ใช้อยู่
            self.allocated_vram -= allocation.vram_allocated
            
            # ตรวจสอบว่าได้ VRAM พอแล้วหรือยัง
            if vram_freed >= vram_needed:
                break
    
    async def release_model_allocation(self, model_id: str) -> bool:
        """
        คืน VRAM ที่ใช้โดยโมเดล
        """
        async with self.lock:
            if model_id in self.model_allocations:
                allocation = self.model_allocations[model_id]
                
                # ลด VRAM ที่ใช้อยู่
                if allocation.location == AllocationLocation.GPU:
                    self.allocated_vram -= allocation.vram_allocated
                    logger.info(f"คืน {allocation.vram_allocated/1024/1024:.1f}MB VRAM จากโมเดล {model_id}")
                
                # ลบการจัดสรร
                del self.model_allocations[model_id]
                return True
            
            return False
    
    async def get_vram_status(self) -> Dict[str, Any]:
        """
        ดูสถานะการใช้ VRAM ปัจจุบัน
        """
        async with self.lock:
            return {
                "total_vram": self.total_vram,
                "allocated_vram": self.allocated_vram,
                "available_vram": self.total_vram - self.allocated_vram,
                "model_allocations": {
                    model_id: {
                        "service": allocation.service_id,
                        "priority": allocation.priority.value,
                        "location": allocation.location.value,
                        "vram": allocation.vram_allocated,
                        "status": allocation.status
                    }
                    for model_id, allocation in self.model_allocations.items()
                }
            }
