"""
Image processing utilities for face analysis APIs
"""

import numpy as np
import cv2
from PIL import Image
import io
from typing import Union, Optional

def process_image_input(image_data: Union[bytes, np.ndarray]) -> np.ndarray:
    """
    Process various image input formats to numpy array
    
    Args:
        image_data: Image data as bytes or numpy array
        
    Returns:
        numpy array in BGR format (for OpenCV compatibility)
    """
    if isinstance(image_data, np.ndarray):
        return image_data
    
    # Convert bytes to numpy array
    if isinstance(image_data, bytes):
        # Try PIL first
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to numpy array (RGB format)
            image_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
            
        except Exception:
            # Fall back to OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_array is None:
                raise ValueError("Failed to decode image data")
            
            return image_array
    
    raise ValueError(f"Unsupported image data type: {type(image_data)}")

def validate_image_format(image_data: bytes, max_size_mb: int = 10) -> bool:
    """
    Validate image format and size
    
    Args:
        image_data: Image data as bytes
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check size
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image too large: {size_mb:.2f}MB > {max_size_mb}MB")
    
    # Try to process image to validate format
    try:
        process_image_input(image_data)
        return True
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")

def resize_image(image: np.ndarray, 
                max_width: int = 1920, 
                max_height: int = 1080,
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image as numpy array
        max_width: Maximum width
        max_height: Maximum height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if width <= max_width and height <= max_height:
        return image
    
    if maintain_aspect:
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = max_width
        new_height = max_height
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def crop_face_region(image: np.ndarray, 
                    bbox: dict, 
                    padding: float = 0.2,
                    target_size: Optional[tuple] = None) -> np.ndarray:
    """
    Crop face region from image with optional padding
    
    Args:
        image: Input image
        bbox: Bounding box dict with x, y, width, height
        padding: Padding factor (0.2 = 20% padding)
        target_size: Optional target size (width, height)
        
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)
    
    # Crop face region
    face_crop = image[y1:y2, x1:x2]
    
    # Resize if target size specified
    if target_size:
        face_crop = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    
    return face_crop

def enhance_face_quality(image: np.ndarray) -> np.ndarray:
    """
    Apply basic enhancement to improve face image quality
    
    Args:
        image: Input face image
        
    Returns:
        Enhanced image
    """
    # Convert to float for processing
    enhanced = image.astype(np.float32) / 255.0
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(enhanced.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply((lab_planes[0] * 255).astype(np.uint8)) / 255.0
        
        # Merge channels
        enhanced = cv2.merge(lab_planes)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((enhanced * 255).astype(np.uint8)) / 255.0
    
    # Apply mild sharpening
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1], 
                      [-1, -1, -1]])
    
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend original and sharpened (mild sharpening)
    enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    # Convert back to uint8
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
    return enhanced

def calculate_image_quality_score(image: np.ndarray) -> float:
    """
    Calculate image quality score based on various metrics
    
    Args:
        image: Input image
        
    Returns:
        Quality score between 0 and 1
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
    
    # Calculate brightness (mean intensity)
    brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray) / 127.5  # Normalize to 0-1
    contrast_score = min(contrast, 1.0)
    
    # Calculate noise level (inverse of smoothness)
    # Use bilateral filter to separate noise from edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    noise_level = np.mean(np.abs(gray.astype(float) - filtered.astype(float))) / 255.0
    noise_score = max(0, 1.0 - noise_level * 5)  # Amplify noise penalty
    
    # Combine scores with weights
    quality_score = (
        sharpness_score * 0.3 +
        brightness_score * 0.25 + 
        contrast_score * 0.25 +
        noise_score * 0.2
    )
    
    return float(np.clip(quality_score, 0.0, 1.0))

def detect_blur_level(image: np.ndarray) -> float:
    """
    Detect blur level in image using Laplacian variance
    
    Args:
        image: Input image
        
    Returns:
        Blur level (0 = very blurry, 1 = sharp)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize to 0-1 range (empirically determined thresholds)
    blur_threshold_low = 100  # Very blurry
    blur_threshold_high = 1000  # Sharp
    
    blur_score = (laplacian_var - blur_threshold_low) / (blur_threshold_high - blur_threshold_low)
    return float(np.clip(blur_score, 0.0, 1.0))

def normalize_face_alignment(image: np.ndarray, landmarks: Optional[list] = None) -> np.ndarray:
    """
    Normalize face alignment based on eye positions (if landmarks available)
    
    Args:
        image: Face image
        landmarks: Optional facial landmarks
        
    Returns:
        Aligned face image
    """
    if landmarks is None or len(landmarks) < 5:
        return image
    
    try:
        # Assume 5-point landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        
        # Calculate angle between eyes
        eye_center = (left_eye + right_eye) / 2
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image to align eyes horizontally
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return aligned
    
    except Exception:
        # Return original image if alignment fails
        return image
