#!/usr/bin/env python3
"""
ระบบ Face Recognition Ensemble Test ขั้นสูง
- ใช้ Ensemble ของ 3 โมเดล: AdaFace (25%), FaceNet (50%), ArcFace (25%)
- ทดสอบกับภาพ boss_01-10 และ night_01-10
- Ultra Quality Enhancement Pipeline
- Advanced performance metrics และ comparison
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# เพิ่ม path เพื่อ import modules
sys.path.append('src')

# เปลี่ยนจาก EnsembleFaceRecognitionService เป็น OptimizedEnsembleFaceRecognitionService
from src.ai_services.face_recognition.ensemble_face_recognition_service import OptimizedEnsembleFaceRecognitionService, EnsembleConfig
from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService, RecognitionConfig
from src.ai_services.face_detection.face_detection_service import FaceDetectionService
from src.ai_services.face_recognition.models import ModelType
from src.ai_services.common.vram_manager import VRAMManager

# Import Ultra Quality Enhancer จาก advanced_test_v13_enhanced.py
from advanced_test_v13_enhanced import UltraQualityEnhancer


class EnsembleTestSystemV15:
    """ระบบทดสอบ Face Recognition Ensemble ขั้นสูง"""
    
    def __init__(self):
        self.setup_logging()
        self.output_dir = Path("output/ensemble_test_v15")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === Ultra Quality Enhancer ===
        self.ultra_enhancer = UltraQualityEnhancer()
        self.logger.info("🚀 Ultra Quality Enhancement initialized")
        
        # === Ensemble Configuration ===
        self.ensemble_config = EnsembleConfig(
            adaface_weight=0.25,   # 25% ตามเอกสาร
            facenet_weight=0.50,   # 50% ตามเอกสาร
            arcface_weight=0.25,   # 25% ตามเอกสาร
            ensemble_threshold=0.20,
            enable_gpu_optimization=True,
            quality_threshold=0.2
        )
        
        # === Individual Model Configuration (สำหรับการเปรียบเทียบ) ===
        self.individual_config = RecognitionConfig(
            similarity_threshold=0.60,
            unknown_threshold=0.55,
            quality_threshold=0.2,
            preferred_model=ModelType.FACENET
        )
        
        # === VRAM Management ===
        vram_config = {
            "total_vram_mb": 12288,  # RTX 3060 12GB
            "reserved_system_mb": 2048,
            "model_vram_estimates": {
                "yolov9c-face": 512 * 1024 * 1024,
                "yolov9e-face": 2048 * 1024 * 1024,
                "yolov11m-face": 2 * 1024 * 1024 * 1024,
                "adaface": 89 * 1024 * 1024,
                "arcface": 249 * 1024 * 1024,
                "facenet": 249 * 1024 * 1024,
            }
        }
        self.vram_manager = VRAMManager(vram_config)
          # === Services ===
        self.ensemble_service = OptimizedEnsembleFaceRecognitionService(self.ensemble_config, self.vram_manager)
        self.individual_service = FaceRecognitionService(self.individual_config, self.vram_manager)
        
        # === Face Detection ===
        detection_config = {
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
        self.detection_service = FaceDetectionService(self.vram_manager, detection_config)
        
        # === Data Storage ===
        self.test_results = {
            'ensemble': defaultdict(list),
            'individual': defaultdict(list),
            'comparison': defaultdict(list)
        }
        
        self.registered_people = {}
        self._initialized = False
        
    def setup_logging(self):
        """ตั้งค่า logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ensemble_test_v15.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def initialize_services(self):
        """Initialize all services"""
        if not self._initialized:
            self.logger.info("🔧 Initializing Ensemble Test System...")
            
            # Initialize services
            ensemble_init = await self.ensemble_service.initialize()
            individual_init = await self.individual_service.initialize()
            detection_init = await self.detection_service.initialize()
            
            if not ensemble_init:
                self.logger.error("❌ Failed to initialize ensemble service")
                return False
                
            if not individual_init:
                self.logger.warning("⚠️ Individual service failed to initialize")
                
            if not detection_init:
                self.logger.error("❌ Failed to initialize detection service")
                return False
                
            self._initialized = True
            self.logger.info("✅ Services initialized successfully")
            return True
        return True

    async def enroll_person_ensemble(self, image_path: str, person_name: str) -> Dict[str, bool]:
        """ลงทะเบียนบุคคลในทั้ง Ensemble และ Individual systems"""
        results = {'ensemble': False, 'individual': False}
        
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"❌ ไม่พบไฟล์: {image_path}")
                return results
                
            # อ่านและปรับปรุงภาพด้วย Ultra Quality Enhancement
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"❌ ไม่สามารถอ่านภาพได้: {image_path}")
                return results
                
            # === ULTRA QUALITY ENHANCEMENT ===
            enhanced_image = self.ultra_enhancer.enhance_image_ultra_quality(image)
            quality_metrics = self.ultra_enhancer.assess_image_quality(enhanced_image)
            
            # ตรวจหาใบหน้า
            detection_result = await self.detection_service.detect_faces(enhanced_image)
            
            if not detection_result.faces:
                self.logger.warning(f"⚠️ ไม่พบใบหน้าใน: {image_path}")
                return results
                
            # ใช้ใบหน้าที่มี confidence สูงสุด
            best_face = max(detection_result.faces, key=lambda f: f.bbox.confidence)
            
            # === ULTRA QUALITY FACE CROPPING ===
            face_crop = self.ultra_enhancer.crop_face_ultra_quality(
                enhanced_image, best_face.bbox, target_size=224
            )
            
            if face_crop.size == 0:
                self.logger.error(f"❌ ไม่สามารถ crop ใบหน้าได้: {image_path}")
                return results
            
            # ลงทะเบียนใน Ensemble system
            ensemble_success = await self.ensemble_service.add_face_to_database(
                person_name, face_crop, 
                metadata={
                    'source_image': image_path,
                    'detection_confidence': best_face.bbox.confidence,
                    'quality_metrics': quality_metrics,
                    'enhancement_applied': True
                }
            )
            results['ensemble'] = ensemble_success
            
            # ลงทะเบียนใน Individual system (สำหรับเปรียบเทียบ)
            individual_embedding = await self.individual_service.extract_embedding(face_crop)
            if individual_embedding is not None:
                await self.individual_service.add_face_to_database(person_name, individual_embedding)
                results['individual'] = True
            
            # เก็บข้อมูลสำหรับการวิเคราะห์
            if person_name not in self.registered_people:
                self.registered_people[person_name] = []
                
            self.registered_people[person_name].append({
                'image_path': image_path,
                'quality_metrics': quality_metrics,
                'detection_confidence': best_face.bbox.confidence,
                'enrollment_time': datetime.now().isoformat(),
                'ensemble_success': ensemble_success,
                'individual_success': results['individual']
            })
            
            status = "✅" if all(results.values()) else "⚠️"
            self.logger.info(f"{status} ลงทะเบียน {person_name} จาก {os.path.basename(image_path)} "
                           f"(Ensemble: {ensemble_success}, Individual: {results['individual']})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ เกิดข้อผิดพลาดในการลงทะเบียน {image_path}: {e}")
            return results

    async def enroll_reference_images(self) -> Dict[str, Any]:
        """ลงทะเบียนภาพอ้างอิงทั้งหมด boss_01-10 และ night_01-10"""
        self.logger.info("📝 เริ่มการลงทะเบียนภาพอ้างอิงด้วย Ensemble System...")
        
        # รายการไฟล์อ้างอิง
        reference_files = []
        
        # Boss images (boss_01 ถึง boss_10)
        for i in range(1, 11):
            reference_files.append((f"test_images/boss_{i:02d}.jpg", "Boss"))
        
        # Night images (night_01 ถึง night_10)  
        for i in range(1, 11):
            reference_files.append((f"test_images/night_{i:02d}.jpg", "Night"))
        
        enrollment_results = {
            'Boss': {'ensemble': 0, 'individual': 0, 'total': 10},
            'Night': {'ensemble': 0, 'individual': 0, 'total': 10},
            'total_files': len(reference_files),
            'successful_enrollments': 0
        }
        
        for image_path, person_name in reference_files:
            results = await self.enroll_person_ensemble(image_path, person_name)
            
            if results['ensemble']:
                enrollment_results[person_name]['ensemble'] += 1
            if results['individual']:
                enrollment_results[person_name]['individual'] += 1
            if any(results.values()):
                enrollment_results['successful_enrollments'] += 1
        
        # แสดงสรุป
        self.logger.info("=" * 60)
        self.logger.info("📊 สรุปการลงทะเบียน Ensemble System:")
        self.logger.info(f"   👤 Boss: Ensemble {enrollment_results['Boss']['ensemble']}/10, "
                        f"Individual {enrollment_results['Boss']['individual']}/10")
        self.logger.info(f"   🌙 Night: Ensemble {enrollment_results['Night']['ensemble']}/10, "
                        f"Individual {enrollment_results['Night']['individual']}/10")
        self.logger.info(f"   📈 รวมทั้งหมด: {enrollment_results['successful_enrollments']}/{enrollment_results['total_files']} ไฟล์")
        self.logger.info(f"   🎯 Ensemble weights: AdaFace 25%, FaceNet 50%, ArcFace 25%")
        
        return enrollment_results

    async def test_recognition_performance(self) -> Dict[str, Any]:
        """ทดสอบประสิทธิภาพการจดจำใบหน้า"""
        self.logger.info("🧪 เริ่มการทดสอบประสิทธิภาพ Ensemble vs Individual...")
        
        # ไฟล์ทดสอบ (ใช้ภาพเดียวกันแต่ทดสอบกับระบบต่างกัน)
        test_files = [
            ("test_images/boss_01.jpg", "Boss"),
            ("test_images/boss_05.jpg", "Boss"),
            ("test_images/boss_10.jpg", "Boss"),
            ("test_images/night_01.jpg", "Night"),
            ("test_images/night_05.jpg", "Night"),
            ("test_images/night_10.jpg", "Night"),
            # เพิ่มไฟล์ที่ไม่ได้ลงทะเบียนเพื่อทดสอบ unknown detection
            ("test_images/boss_11.jpg", "Unknown"),
            ("test_images/boss_glass02.jpg", "Unknown"),
        ]
        
        results = {
            'ensemble_results': [],
            'individual_results': [],
            'performance_comparison': {
                'ensemble': {'correct': 0, 'total': 0, 'processing_times': []},
                'individual': {'correct': 0, 'total': 0, 'processing_times': []}
            }
        }
        
        for image_path, expected_identity in test_files:
            if not os.path.exists(image_path):
                self.logger.warning(f"⚠️ ไม่พบไฟล์: {image_path}")
                continue
                
            # อ่านและปรับปรุงภาพ
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            enhanced_image = self.ultra_enhancer.enhance_image_ultra_quality(image)
              # ตรวจหาใบหน้า
            detection_result = await self.detection_service.detect_faces(enhanced_image)
            if not detection_result.faces:
                continue
                
            best_face = max(detection_result.faces, key=lambda f: f.bbox.confidence)
            face_crop = self.ultra_enhancer.crop_face_ultra_quality(
                enhanced_image, best_face.bbox, target_size=224
            )
            
            # ตรวจสอบและแก้ไข face_crop
            if face_crop is None or face_crop.size == 0:
                self.logger.warning(f"⚠️ Face crop failed for {image_path}")
                continue
                
            # Ensure face_crop is numpy array
            if not isinstance(face_crop, np.ndarray):
                self.logger.warning(f"⚠️ Face crop is not numpy array for {image_path}: {type(face_crop)}")
                continue
            
            # ทดสอบด้วย Ensemble
            ensemble_result = await self.ensemble_service.recognize_face(face_crop)
            ensemble_identity = ensemble_result.best_match.identity_id if ensemble_result.best_match else "Unknown"
            ensemble_correct = (ensemble_identity == expected_identity)
            
            results['ensemble_results'].append({
                'image_path': image_path,
                'expected': expected_identity,
                'predicted': ensemble_identity,
                'confidence': ensemble_result.confidence,
                'similarity': ensemble_result.best_match.similarity if ensemble_result.best_match else 0.0,
                'correct': ensemble_correct,
                'processing_time': ensemble_result.processing_time
            })
            
            results['performance_comparison']['ensemble']['total'] += 1
            if ensemble_correct:
                results['performance_comparison']['ensemble']['correct'] += 1
            results['performance_comparison']['ensemble']['processing_times'].append(ensemble_result.processing_time)
            
            # ทดสอบด้วย Individual model
            individual_result = await self.individual_service.recognize_face(face_crop)
            individual_identity = individual_result.best_match.identity_id if individual_result.best_match else "Unknown"
            individual_correct = (individual_identity == expected_identity)
            
            results['individual_results'].append({
                'image_path': image_path,
                'expected': expected_identity,
                'predicted': individual_identity,
                'confidence': individual_result.confidence,
                'similarity': individual_result.best_match.similarity if individual_result.best_match else 0.0,
                'correct': individual_correct,
                'processing_time': individual_result.processing_time
            })
            
            results['performance_comparison']['individual']['total'] += 1
            if individual_correct:
                results['performance_comparison']['individual']['correct'] += 1
            results['performance_comparison']['individual']['processing_times'].append(individual_result.processing_time)
            
            self.logger.info(f"📝 {os.path.basename(image_path)}: "
                           f"Ensemble={ensemble_identity}({ensemble_result.confidence:.3f}), "
                           f"Individual={individual_identity}({individual_result.confidence:.3f})")
        
        return results

    def generate_performance_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างรายงานประสิทธิภาพ"""
        try:
            report = {
                'summary': {},
                'detailed_analysis': {},
                'recommendations': []
            }
            
            # คำนวณ accuracy
            ensemble_stats = test_results['performance_comparison']['ensemble']
            individual_stats = test_results['performance_comparison']['individual']
            
            ensemble_accuracy = ensemble_stats['correct'] / max(ensemble_stats['total'], 1)
            individual_accuracy = individual_stats['correct'] / max(individual_stats['total'], 1)
            
            # คำนวณ processing time
            ensemble_avg_time = np.mean(ensemble_stats['processing_times']) if ensemble_stats['processing_times'] else 0
            individual_avg_time = np.mean(individual_stats['processing_times']) if individual_stats['processing_times'] else 0
            
            report['summary'] = {
                'ensemble_accuracy': ensemble_accuracy,
                'individual_accuracy': individual_accuracy,
                'accuracy_improvement': ensemble_accuracy - individual_accuracy,
                'ensemble_avg_processing_time': ensemble_avg_time,
                'individual_avg_processing_time': individual_avg_time,
                'processing_time_difference': ensemble_avg_time - individual_avg_time,
                'total_tests': ensemble_stats['total']
            }
            
            # Detailed analysis
            ensemble_confidences = [r['confidence'] for r in test_results['ensemble_results']]
            individual_confidences = [r['confidence'] for r in test_results['individual_results']]
            
            report['detailed_analysis'] = {
                'ensemble_confidence_stats': {
                    'mean': np.mean(ensemble_confidences) if ensemble_confidences else 0,
                    'std': np.std(ensemble_confidences) if ensemble_confidences else 0,
                    'min': np.min(ensemble_confidences) if ensemble_confidences else 0,
                    'max': np.max(ensemble_confidences) if ensemble_confidences else 0
                },
                'individual_confidence_stats': {
                    'mean': np.mean(individual_confidences) if individual_confidences else 0,
                    'std': np.std(individual_confidences) if individual_confidences else 0,
                    'min': np.min(individual_confidences) if individual_confidences else 0,
                    'max': np.max(individual_confidences) if individual_confidences else 0
                }
            }
            
            # Recommendations
            if ensemble_accuracy > individual_accuracy:
                report['recommendations'].append("✅ Ensemble system แสดงประสิทธิภาพที่ดีกว่า individual model")
            else:
                report['recommendations'].append("⚠️ Individual model แสดงประสิทธิภาพที่ดีกว่า ensemble system")
                
            if ensemble_avg_time > individual_avg_time * 1.5:
                report['recommendations'].append("⚠️ Ensemble system ใช้เวลาประมวลผลมากกว่า 50% - พิจารณา optimization")
            else:
                report['recommendations'].append("✅ เวลาประมวลผลของ ensemble system อยู่ในเกณฑ์ที่ยอมรับได้")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance report: {e}")
            return {'error': str(e)}

    def save_results(self, enrollment_results: Dict[str, Any], test_results: Dict[str, Any], 
                    performance_report: Dict[str, Any]):
        """บันทึกผลลัพธ์"""
        try:
            # บันทึกผลลัพธ์หลัก
            results_file = self.output_dir / f"ensemble_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            all_results = {
                'test_info': {
                    'timestamp': datetime.now().isoformat(),
                    'ensemble_config': {
                        'adaface_weight': self.ensemble_config.adaface_weight,
                        'facenet_weight': self.ensemble_config.facenet_weight,
                        'arcface_weight': self.ensemble_config.arcface_weight,
                        'ensemble_threshold': self.ensemble_config.ensemble_threshold
                    },
                    'models_used': ['AdaFace IR101', 'FaceNet VGGFace2', 'ArcFace R100']
                },
                'enrollment_results': enrollment_results,
                'test_results': test_results,
                'performance_report': performance_report,
                'ensemble_statistics': self.ensemble_service.get_statistics()
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"💾 ผลลัพธ์บันทึกแล้วที่: {results_file}")
            
            # สร้างรายงาน markdown
            self.generate_markdown_report(all_results)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")

    def generate_markdown_report(self, results: Dict[str, Any]):
        """สร้างรายงาน markdown"""
        try:
            report_file = self.output_dir / f"ensemble_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# Face Recognition Ensemble Test Report\n\n")
                f.write(f"Generated: {results['test_info']['timestamp']}\n\n")
                
                # Configuration
                f.write("## Configuration\n\n")
                config = results['test_info']['ensemble_config']
                f.write(f"- **AdaFace Weight**: {config['adaface_weight']} (25%)\n")
                f.write(f"- **FaceNet Weight**: {config['facenet_weight']} (50%)\n")
                f.write(f"- **ArcFace Weight**: {config['arcface_weight']} (25%)\n")
                f.write(f"- **Ensemble Threshold**: {config['ensemble_threshold']}\n")
                f.write(f"- **Models**: {', '.join(results['test_info']['models_used'])}\n\n")
                
                # Enrollment Results
                f.write("## Enrollment Results\n\n")
                enrollment = results['enrollment_results']
                f.write(f"- **Total Files**: {enrollment['total_files']}\n")
                f.write(f"- **Successful Enrollments**: {enrollment['successful_enrollments']}\n")
                f.write(f"- **Boss Images**: Ensemble {enrollment['Boss']['ensemble']}/10, Individual {enrollment['Boss']['individual']}/10\n")
                f.write(f"- **Night Images**: Ensemble {enrollment['Night']['ensemble']}/10, Individual {enrollment['Night']['individual']}/10\n\n")
                
                # Performance Summary
                f.write("## Performance Summary\n\n")
                summary = results['performance_report']['summary']
                f.write(f"- **Ensemble Accuracy**: {summary['ensemble_accuracy']:.3f} ({summary['ensemble_accuracy']*100:.1f}%)\n")
                f.write(f"- **Individual Accuracy**: {summary['individual_accuracy']:.3f} ({summary['individual_accuracy']*100:.1f}%)\n")
                f.write(f"- **Accuracy Improvement**: {summary['accuracy_improvement']:.3f} ({summary['accuracy_improvement']*100:.1f}%)\n")
                f.write(f"- **Ensemble Avg Processing Time**: {summary['ensemble_avg_processing_time']:.3f}s\n")
                f.write(f"- **Individual Avg Processing Time**: {summary['individual_avg_processing_time']:.3f}s\n")
                f.write(f"- **Processing Time Difference**: {summary['processing_time_difference']:.3f}s\n\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                for rec in results['performance_report']['recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
                
                # Ensemble Statistics
                f.write("## Ensemble Statistics\n\n")
                stats = results['ensemble_statistics']
                f.write(f"- **Total Extractions**: {stats['total_extractions']}\n")
                f.write(f"- **Total Recognitions**: {stats['total_recognitions']}\n")
                f.write(f"- **Successful Recognitions**: {stats['successful_recognitions']}\n")
                f.write(f"- **Database Size**: {stats['database_size']} persons\n")
                f.write(f"- **Total Embeddings**: {stats['total_embeddings']}\n\n")
                
                # Model Success Rates
                f.write("### Model Success Rates\n\n")
                for model, model_stats in stats['model_success_rates'].items():
                    f.write(f"- **{model.upper()}**: {model_stats['success']}/{model_stats['total']} "
                           f"({model_stats.get('success_rate', 0)*100:.1f}%)\n")
                
            self.logger.info(f"📄 รายงาน markdown บันทึกแล้วที่: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating markdown report: {e}")

    async def run_full_test(self):
        """รันการทดสอบแบบเต็ม"""
        try:
            self.logger.info("🚀 เริ่มการทดสอบ Face Recognition Ensemble System V15")
            self.logger.info("=" * 80)
            
            # Initialize services
            if not await self.initialize_services():
                self.logger.error("❌ Failed to initialize services")
                return
            
            # Enroll reference images
            enrollment_results = await self.enroll_reference_images()
            
            # Test recognition performance
            test_results = await self.test_recognition_performance()
            
            # Generate performance report
            performance_report = self.generate_performance_report(test_results)
            
            # Display summary
            self.logger.info("=" * 80)
            self.logger.info("📊 สรุปผลการทดสอบ Ensemble System:")
            
            summary = performance_report['summary']
            self.logger.info(f"   🎯 Ensemble Accuracy: {summary['ensemble_accuracy']*100:.1f}%")
            self.logger.info(f"   🔄 Individual Accuracy: {summary['individual_accuracy']*100:.1f}%") 
            self.logger.info(f"   📈 Improvement: {summary['accuracy_improvement']*100:.1f}%")
            self.logger.info(f"   ⏱️ Ensemble Time: {summary['ensemble_avg_processing_time']:.3f}s")
            self.logger.info(f"   ⏱️ Individual Time: {summary['individual_avg_processing_time']:.3f}s")
            
            # Get ensemble statistics
            # แก้ไขการเรียกใช้ method สำหรับ OptimizedEnsembleFaceRecognitionService
            ensemble_stats = self.ensemble_service.get_enhanced_statistics()
            self.logger.info(f"   🔧 Model Success Rates:")
            for model, stats in ensemble_stats['model_success_rates'].items():
                rate = stats.get('success_rate', 0) * 100
                self.logger.info(f"      - {model.upper()}: {rate:.1f}%")
            
            # Save results
            self.save_results(enrollment_results, test_results, performance_report)
            
            self.logger.info("=" * 80)
            self.logger.info("✅ การทดสอบ Ensemble System เสร็จสมบูรณ์!")
            
        except Exception as e:
            self.logger.error(f"❌ Error during full test: {e}")
            raise


async def main():
    """ฟังก์ชันหลัก"""
    try:
        # สร้างระบบทดสอบ
        test_system = EnsembleTestSystemV15()
        
        # รันการทดสอบแบบเต็ม
        await test_system.run_full_test()
        
    except Exception as e:
        logging.error(f"❌ การทดสอบล้มเหลว: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
