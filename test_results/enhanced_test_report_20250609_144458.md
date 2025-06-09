# Face Social AI Services - Enhanced Production Test Report
Generated: 2025-06-09 14:44:58

## Executive Summary

## System Information

- **Platform**: Windows-10-10.0.26100-SP0
- **Python**: 3.11.13
- **PyTorch**: 2.5.1
- **CUDA Available**: ✅
- **ONNX Runtime**: 1.22.0

## GPU Information

### GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU

- **Memory**: 6.0 GB
- **Compute Capability**: 8.6
- **Multiprocessors**: 30

## Model Availability

**Overall**: 10/10 (100.0%)

### Face Detection

| Model | Status | Size (MB) |
|-------|--------|-----------|
| yolov9c-face-lindevs.onnx | ✅ | 80.98 |
| yolov9e-face-lindevs.onnx | ✅ | 203.39 |
| yolov11m-face.pt | ✅ | 38.62 |

### Face Recognition

| Model | Status | Size (MB) |
|-------|--------|-----------|
| adaface_ir101.onnx | ✅ | 248.62 |
| arcface_r100.onnx | ✅ | 248.62 |
| facenet_vggface2.onnx | ✅ | 89.61 |

### Anti Spoofing

| Model | Status | Size (MB) |
|-------|--------|-----------|
| AntiSpoofing_bin_1.5_128.onnx | ✅ | 1.81 |
| AntiSpoofing_print-replay_1.5_128.onnx | ✅ | 1.81 |

### Deepfake Detection

| Model | Status | Size (MB) |
|-------|--------|-----------|
| model.onnx | ✅ | 44.22 |

### Gender Age

| Model | Status | Size (MB) |
|-------|--------|-----------|
| genderage.onnx | ✅ | 1.26 |

## GPU Capabilities

### Memory Information

- **Total**: 6143.5 MB
- **Free**: 6143.5 MB
- **Allocated**: 0.0 MB

### Performance Test

- **Matrix Multiplication**: 0.1223 seconds

## Test Results Summary

- **Total Execution Time**: 1.36 seconds
- **Errors**: 0
