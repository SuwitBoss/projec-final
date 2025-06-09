#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Embedding Face Recognition Test
ใช้หลายภาพสำหรับสร้าง embedding แต่ละคน เพื่อเพิ่มความแม่นยำ
- Boss: boss_01.jpg ถึง boss_05.jpg (5 ภาพ)
- Night: night_01.jpg ถึง night_03.jpg (3 ภาพ)
"""

import os
import sys
import asyncio
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main test function"""
    try:
        logger.info("🚀 Starting Multi-Embedding Face Recognition Test")
          # Import services after path setup
        from ai_services.common.vram_manager import VRAMManager
        from ai_services.face_detection.face_detection_service import FaceDetectionService
        from ai_services.face_recognition.face_recognition_service import FaceRecognitionService, RecognitionConfig
        from ai_services.face_recognition.models import ModelType
        from ai_services.face_detection.utils import BoundingBox
        
        # VRAM configuration
        vram_config = {
            "model_vram_estimates": {
                "YOLOv9e": 2048 * 1024 * 1024,    # 2GB
                "YOLOv11m": 1024 * 1024 * 1024,   # 1GB
                "ADAFACE": 249 * 1024 * 1024,     # 249MB
                "ARCFACE": 249 * 1024 * 1024,     # 249MB
                "FACENET": 89 * 1024 * 1024,      # 89MB
                "default": 512 * 1024 * 1024      # 512MB
            },
            "reserved_vram": 1024 * 1024 * 1024,  # 1GB reserved
            "priority_threshold": 0.8,
            "use_smart_allocation": True,
            "enable_model_unloading": True
        }
        
        # Initialize services
        logger.info("🔧 Initializing AI Services...")
        
        # Initialize VRAM Manager
        vram_manager = VRAMManager(vram_config)
        
        # Detection service config
        detection_config = {
            "models": {
                "yolov9c-face": {
                    "model_path": "model/face-detection/yolov9c-face.onnx",
                    "confidence_threshold": 0.3,
                    "iou_threshold": 0.5,
                    "input_size": (640, 640)
                },
                "yolov9e-face": {
                    "model_path": "model/face-detection/yolov9e-face.onnx", 
                    "confidence_threshold": 0.3,
                    "iou_threshold": 0.5,
                    "input_size": (640, 640)
                },
                "yolov11m-face": {
                    "model_path": "model/face-detection/yolov11m-face.pt",
                    "confidence_threshold": 0.3,
                    "iou_threshold": 0.5,
                    "input_size": (640, 640)
                }
            },
            "quality_threshold": 50.0,
            "detection_method": "enhanced_intelligent"
        }        # Initialize detection service
        detection_service = FaceDetectionService(vram_manager, detection_config)
        await detection_service.initialize()
          # Recognition service config
        recognition_config = RecognitionConfig(
            similarity_threshold=0.5,
            max_faces=10,
            quality_threshold=0.3,
            auto_model_selection=False,
            preferred_model=ModelType.FACENET,
            enable_quality_assessment=True
        )# Initialize recognition service
        recognition_service = FaceRecognitionService(recognition_config, vram_manager)
        await recognition_service.initialize()
        
        logger.info("✅ All AI Services initialized successfully")
        
        # Build multi-embedding face gallery
        logger.info("🖼️ Building Multi-Embedding Face Gallery...")
        gallery_success = await build_multi_embedding_gallery(
            recognition_service, detection_service, project_root / "test_images"
        )
        
        if not gallery_success:
            logger.error("❌ Failed to build face gallery")
            return
        
        logger.info("✅ Multi-Embedding Face Gallery built successfully")
        
        # Get all test images
        test_images_dir = project_root / "test_images"
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(test_images_dir.glob(ext))
        
        # Sort images for consistent processing order
        image_files.sort()
        
        logger.info(f"🔍 Found {len(image_files)} test images")
        
        # Process each image
        results = {}
        total_faces_detected = 0
        total_faces_recognized = 0
        processing_times = []
        
        for i, image_path in enumerate(image_files, 1):
            try:
                logger.info(f"📸 Processing [{i:2d}/{len(image_files)}]: {image_path.name}")
                
                start_time = time.time()
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"❌ Failed to load image: {image_path.name}")
                    continue
                
                # Detect faces
                detection_result = await detection_service.detect_faces(image)
                faces_detected = len(detection_result.faces)
                total_faces_detected += faces_detected
                
                # Process each detected face
                face_results = []
                recognized_persons = []
                
                for j, face in enumerate(detection_result.faces):
                    # Extract face crop
                    face_crop = extract_face_crop(image, face.bbox)
                    if face_crop is None:
                        continue
                    
                    # Recognize face
                    recognition_result = await recognition_service.recognize_face(face_crop)
                    
                    face_info = {
                        'face_id': f"face_{j:02d}",
                        'bbox': {
                            'x1': float(face.bbox.x1),
                            'y1': float(face.bbox.y1),
                            'x2': float(face.bbox.x2),
                            'y2': float(face.bbox.y2)
                        },
                        'confidence': float(face.bbox.confidence),
                        'quality_score': float(face.quality_score),
                        'recognition': {
                            'best_match': None,
                            'confidence': 0.0,
                            'all_matches': []
                        }
                    }
                    
                    # Process recognition results
                    if recognition_result.best_match:
                        face_info['recognition']['best_match'] = recognition_result.best_match.person_id
                        face_info['recognition']['confidence'] = float(recognition_result.best_match.confidence)
                        recognized_persons.append(recognition_result.best_match.person_id)
                        total_faces_recognized += 1
                    
                    # Add all matches
                    for match in recognition_result.matches:
                        face_info['recognition']['all_matches'].append({
                            'person_id': match.person_id,
                            'confidence': float(match.confidence)
                        })
                    
                    face_results.append(face_info)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Store results
                results[image_path.name] = {
                    'faces_detected': faces_detected,
                    'faces_recognized': len([f for f in face_results if f['recognition']['best_match']]),
                    'recognized_persons': list(set(recognized_persons)),  # Unique persons
                    'processing_time': float(processing_time),
                    'detection_model': detection_result.model_used,
                    'faces': face_results
                }
                
                # Log summary
                recognized_summary = ", ".join(set(recognized_persons)) if recognized_persons else "None"
                logger.info(
                    f"   ✅ {faces_detected} faces detected, "
                    f"{len([f for f in face_results if f['recognition']['best_match']])} recognized "
                    f"[{recognized_summary}] in {processing_time:.3f}s"
                )
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_path.name}: {e}")
                results[image_path.name] = {
                    'error': str(e),
                    'faces_detected': 0,
                    'faces_recognized': 0,
                    'recognized_persons': [],
                    'processing_time': 0.0
                }
        
        # Generate comprehensive analysis report
        await generate_analysis_report(results, total_faces_detected, total_faces_recognized, processing_times)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = project_root / "test_results" / f"multi_embedding_face_test_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Detailed results saved to: {results_file}")
        
        # Create visual results
        await create_visual_results(results, project_root / "test_images", project_root / "output")
        
        logger.info("🎉 Multi-Embedding Face Recognition Test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def build_multi_embedding_gallery(recognition_service, detection_service, test_images_dir):
    """Build face gallery with multiple embeddings per person"""
    try:
        # Reference images for each person (multiple per person)
        reference_images = {
            'boss': [f'boss_0{i}.jpg' for i in range(1, 6)],  # boss_01.jpg ถึง boss_05.jpg
            'night': [f'night_0{i}.jpg' for i in range(1, 4)]  # night_01.jpg ถึง night_03.jpg
        }
        
        total_embeddings = 0
        
        for person_id, image_names in reference_images.items():
            logger.info(f"📝 Creating embeddings for {person_id.upper()}...")
            person_embeddings = 0
            
            for image_name in image_names:
                image_path = test_images_dir / image_name
                
                if not image_path.exists():
                    logger.warning(f"⚠️ Reference image not found: {image_name}")
                    continue
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"❌ Failed to load reference image: {image_name}")
                    continue
                
                # Detect faces in reference image
                detection_result = await detection_service.detect_faces(image)
                
                if not detection_result.faces:
                    logger.error(f"❌ No faces detected in reference image: {image_name}")
                    continue
                
                # Use the largest/most confident face
                best_face = max(detection_result.faces, key=lambda f: f.bbox.confidence * f.quality_score)
                
                # Extract face crop
                face_crop = extract_face_crop(image, best_face.bbox)
                if face_crop is None:
                    logger.error(f"❌ Failed to extract face crop from: {image_name}")
                    continue
                
                # Add to recognition database
                success = await recognition_service.add_face_to_database(
                    person_id, 
                    face_crop, 
                    metadata={'reference_image': image_name, 'embedding_index': person_embeddings}
                )
                
                if success:
                    person_embeddings += 1
                    total_embeddings += 1
                    logger.info(f"   ✅ Added embedding {person_embeddings} for {person_id} from {image_name}")
                else:
                    logger.error(f"❌ Failed to add embedding for {person_id} from {image_name}")
            
            logger.info(f"✅ Created {person_embeddings} embeddings for {person_id.upper()}")
        
        logger.info(f"🎯 Total embeddings created: {total_embeddings}")
        return total_embeddings > 0
        
    except Exception as e:
        logger.error(f"❌ Error building multi-embedding gallery: {e}")
        return False

def extract_face_crop(image, bbox):
    """Extract face crop from image using bounding box"""
    try:
        height, width = image.shape[:2]
        
        # Get coordinates with bounds checking
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(width, int(bbox.x2))
        y2 = min(height, int(bbox.y2))
        
        # Validate crop area
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract face crop
        face_crop = image[y1:y2, x1:x2]
        
        # Ensure minimum size
        if face_crop.shape[0] < 32 or face_crop.shape[1] < 32:
            return None
        
        return face_crop
        
    except Exception as e:
        logger.error(f"Error extracting face crop: {e}")
        return None

async def generate_analysis_report(results, total_faces_detected, total_faces_recognized, processing_times):
    """Generate comprehensive analysis report"""
    logger.info("\n" + "="*80)
    logger.info("📊 MULTI-EMBEDDING FACE RECOGNITION ANALYSIS REPORT")
    logger.info("="*80)
    
    # Overall statistics
    total_images = len(results)
    successful_images = len([r for r in results.values() if 'error' not in r])
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    logger.info(f"📈 OVERALL STATISTICS:")
    logger.info(f"   Total Images: {total_images}")
    logger.info(f"   Successfully Processed: {successful_images}")
    logger.info(f"   Total Faces Detected: {total_faces_detected}")
    logger.info(f"   Total Faces Recognized: {total_faces_recognized}")
    logger.info(f"   Recognition Rate: {(total_faces_recognized/total_faces_detected*100):.1f}%" if total_faces_detected > 0 else "   Recognition Rate: 0.0%")
    logger.info(f"   Average Processing Time: {avg_processing_time:.3f}s per image")
    
    # Category analysis
    categories = {
        'boss_single': [k for k in results.keys() if k.startswith('boss_') and 'group' not in k and 'glass' not in k],
        'boss_group': [k for k in results.keys() if k.startswith('boss_group')],
        'boss_glasses': [k for k in results.keys() if k.startswith('boss_glass')],
        'night_single': [k for k in results.keys() if k.startswith('night_') and 'group' not in k],
        'night_group': [k for k in results.keys() if k.startswith('night_group')],
        'spoofing': [k for k in results.keys() if k.startswith('spoofing_')],
        'face_swap': [k for k in results.keys() if k.startswith('face-swap')]
    }
    
    logger.info(f"\n📋 CATEGORY ANALYSIS:")
    for category, images in categories.items():
        if not images:
            continue
        
        category_results = {k: results[k] for k in images if k in results}
        faces_detected = sum(r.get('faces_detected', 0) for r in category_results.values())
        faces_recognized = sum(r.get('faces_recognized', 0) for r in category_results.values())
        
        # Person identification analysis
        boss_detections = 0
        night_detections = 0
        for r in category_results.values():
            if 'boss' in r.get('recognized_persons', []):
                boss_detections += 1
            if 'night' in r.get('recognized_persons', []):
                night_detections += 1
        
        logger.info(f"   {category.upper()}:")
        logger.info(f"     Images: {len(images)}")
        logger.info(f"     Faces Detected: {faces_detected}")
        logger.info(f"     Faces Recognized: {faces_recognized}")
        logger.info(f"     Boss Identified: {boss_detections} images")
        logger.info(f"     Night Identified: {night_detections} images")
    
    # Person identification summary
    logger.info(f"\n👤 PERSON IDENTIFICATION SUMMARY:")
    boss_appearances = []
    night_appearances = []
    
    for image_name, result in results.items():
        recognized_persons = result.get('recognized_persons', [])
        if 'boss' in recognized_persons:
            boss_appearances.append(image_name)
        if 'night' in recognized_persons:
            night_appearances.append(image_name)
    
    logger.info(f"   BOSS identified in {len(boss_appearances)} images:")
    for img in sorted(boss_appearances):
        faces_with_boss = [f for f in results[img].get('faces', []) if f['recognition']['best_match'] == 'boss']
        max_confidence = max([f['recognition']['confidence'] for f in faces_with_boss]) if faces_with_boss else 0
        logger.info(f"     - {img} (confidence: {max_confidence:.3f})")
    
    logger.info(f"   NIGHT identified in {len(night_appearances)} images:")
    for img in sorted(night_appearances):
        faces_with_night = [f for f in results[img].get('faces', []) if f['recognition']['best_match'] == 'night']
        max_confidence = max([f['recognition']['confidence'] for f in faces_with_night]) if faces_with_night else 0
        logger.info(f"     - {img} (confidence: {max_confidence:.3f})")
    
    logger.info("="*80)

async def create_visual_results(results, test_images_dir, output_dir):
    """Create visual results with bounding boxes and labels"""
    try:
        output_dir = output_dir / "multi_embedding_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎨 Creating visual results...")
        
        # Color mapping for persons
        colors = {
            'boss': (0, 255, 0),      # Green
            'night': (255, 0, 0),     # Blue
            'unknown': (128, 128, 128) # Gray
        }
        
        processed_count = 0
        
        for image_name, result in results.items():
            if 'error' in result:
                continue
                
            # Load original image
            image_path = test_images_dir / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Draw bounding boxes and labels
            for face in result.get('faces', []):
                bbox = face['bbox']
                person_id = face['recognition']['best_match']
                confidence = face['recognition']['confidence']
                
                # Get coordinates
                x1, y1 = int(bbox['x1']), int(bbox['y1'])
                x2, y2 = int(bbox['x2']), int(bbox['y2'])
                
                # Choose color
                color = colors.get(person_id, colors['unknown'])
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                if person_id:
                    label = f"{person_id}: {confidence:.3f}"
                else:
                    label = "Unknown"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add summary info
            summary = f"Faces: {result['faces_detected']}, Recognized: {result['faces_recognized']}"
            cv2.putText(image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save result image
            output_path = output_dir / f"result_{image_name}"
            cv2.imwrite(str(output_path), image)
            processed_count += 1
        
        logger.info(f"✅ Created {processed_count} visual result images in {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error creating visual results: {e}")

if __name__ == "__main__":
    asyncio.run(main())
