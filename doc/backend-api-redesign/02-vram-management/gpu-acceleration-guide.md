# GPU Acceleration for AI Services

This document outlines the GPU acceleration implementation for the facial recognition system and provides guidance on optimizing GPU memory usage, performance benchmarking, and troubleshooting.

## Overview

Our facial recognition system now fully utilizes GPU acceleration through ONNX Runtime's CUDA Execution Provider. This significantly improves inference speed for:

- Face detection (YOLOv10n model)
- Face recognition (ArcFace and AdaFace models)
- Deepfake detection (Xception model)

## Performance Improvements

Based on benchmark tests, we've observed the following performance improvements:

| AI Service | CPU Time (avg) | GPU Time (avg) | Speedup |
|------------|---------------|---------------|---------|
| Face Detection | ~80-120ms | ~15-25ms | 4-5x |
| Face Recognition | ~150-200ms | ~20-30ms | 5-7x |
| Deepfake Detection | ~300-400ms | ~60-80ms | 4-5x |

*Note: Actual performance may vary based on hardware, image size, and number of faces.*

## GPU Memory Management

ONNX Runtime allows fine-grained control over GPU memory usage. We've implemented the following optimizations:

### Memory Allocation Strategies

Each AI service uses a customized GPU memory allocation strategy:

```python
providers.append(('CUDAExecutionProvider', {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}))
```

Key parameters:

- **device_id**: Specifies which GPU to use (0 = first GPU)
- **arena_extend_strategy**: Controls how memory is allocated when more is needed
- **gpu_mem_limit**: Maximum memory usage per model (customized for each service)
- **cudnn_conv_algo_search**: Algorithm search strategy (EXHAUSTIVE provides best performance)
- **do_copy_in_default_stream**: Ensures synchronization between host and device

### Memory Allocation by Service

| Service | GPU Memory Limit | Rationale |
|---------|-----------------|-----------|
| Face Detection | 2GB | YOLOv10n is relatively small but requires enough memory for batch processing |
| Face Recognition | 2GB | Embedding models require significant memory for feature extraction |
| Deepfake Detection | 1GB | Xception model is optimized to require less memory |

## Optimization Techniques

### 1. Model Quantization

For further memory optimization, models can be quantized:

- **FP16 Quantization**: Reduces precision from 32-bit to 16-bit floats, cutting memory usage by ~50% with minimal accuracy loss
- **INT8 Quantization**: Further reduces to 8-bit integers, reducing memory by ~75% but with potential accuracy impact

### 2. Graph Optimization

ONNX Runtime provides several graph optimization levels:

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### 3. Provider Priority

Our system prioritizes providers in this order:

1. CUDA (NVIDIA GPUs)
2. DirectML (Windows-only, for AMD/Intel GPUs)
3. CPU (fallback)

### 4. Session Reuse

For efficiency, inference sessions are loaded once at startup and reused for all requests.

## Implementation Details

### 1. Model Loading with GPU Support

All AI services follow this pattern for GPU acceleration:

```python
def _load_model(self):
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = []
        
        # Try CUDA provider first
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
            
        # Fallback to CPU
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
    except Exception as e:
        # Error handling and fallback
```

### 2. Efficient Data Processing

To minimize CPU-GPU data transfers:

- Preprocessing is done on CPU
- Data is transferred to GPU once
- Processing is done on GPU
- Results are transferred back to CPU once

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Symptoms: CUDA out of memory errors
   - Solutions: 
     - Reduce batch size
     - Lower `gpu_mem_limit` parameter
     - Try FP16 quantization

2. **Model Loading Failures**
   - Symptoms: "Failed to load model" errors
   - Solutions:
     - Check CUDA/GPU driver compatibility
     - Verify model format is compatible with ONNX Runtime
     - Try CPU fallback

3. **Performance Not Improving**
   - Symptoms: GPU not faster than CPU
   - Solutions:
     - Check if model is actually using GPU (check session providers)
     - Increase `cudnn_conv_algo_search` to EXHAUSTIVE
     - Profile data preprocessing steps (might be CPU bottlenecked)

## Performance Monitoring

A benchmark script is available to measure and compare CPU vs GPU performance:

```bash
python benchmark_gpu_performance.py --image test_image.jpg --output benchmark_results
```

This script:
1. Runs each AI service on both CPU and GPU
2. Measures average inference time
3. Calculates speedup ratio
4. Generates performance charts
5. Saves results as JSON for analysis

## Future Improvements

1. **Model Optimization**: Implement automatic model quantization
2. **Multi-GPU Support**: Add load balancing across multiple GPUs
3. **Dynamic Memory Management**: Adjust memory limits based on system load
4. **Batch Processing**: Optimize for processing multiple images simultaneously
5. **Model Caching**: Implement intelligent model loading/unloading based on usage patterns

## References

- [ONNX Runtime GPU Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [Model Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
