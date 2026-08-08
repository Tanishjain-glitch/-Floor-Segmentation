# Fast-SCNN Floor Segmentation for Edge Deployment

Real-time semantic floor segmentation using Fast-SCNN, with deployment through
PyTorch, ONNX Runtime, and Intel OpenVINO.

The system takes an RGB image/video stream and produces a binary segmentation
mask identifying the traversable floor region.

---

## 1. Overview

This project implements a lightweight semantic segmentation pipeline for
indoor floor detection.

The model is trained on a custom floor segmentation dataset with one class:

- 1: Floor

The trained Fast-SCNN model is exported to ONNX and converted to OpenVINO IR
for CPU-first edge deployment.

Deployment targets:

- PyTorch + CUDA
- ONNX Runtime
- OpenVINO CPU
- OpenVINO NPU fallback logic

---

## 2. Architecture

Fast-SCNN was selected because it is specifically designed for fast semantic
segmentation while maintaining a lightweight architecture.

Pipeline:

    RGB Image / Video
            |
            v
      Preprocessing
            |
            v
        Fast-SCNN
            |
            v
      Floor Prediction
            |
            v
    Post-processing
            |
            v
      Floor Overlay

Deployment pipeline:

    PyTorch (.pth)
          |
          v
    ONNX (.onnx)
          |
          v
    OpenVINO IR
      (.xml + .bin)
          |
          v
    CPU / NPU deployment

---

## 3. Dataset

A custom floor segmentation dataset was used.

Dataset structure:

    datasets/
    └── floor/
        ├── train/
        │   ├── images/
        │   └── masks/
        │
        └── val/
            ├── images/
            └── masks/

Each RGB image has a corresponding binary segmentation mask.

Mask convention:

    0 = Background
    1 = Floor

---

## 4. Training

The model was trained using:

- Model: Fast-SCNN
- Number of classes: 2
- Input resolution: 512 x 512
- Batch size: 8
- Optimizer: SGD
- Initial learning rate: 0.001
- Momentum: 0.9
- Weight decay: 0.0001
- Epochs: 100

Training command:

    python train.py --dataset floor

The best trained model is:

    weights/fast_scnn_floor_best_model.pth

---

## 5. Validation Results

The trained model achieved approximately:

- Pixel Accuracy: 94.9%
- mIoU: 87.8%

These results demonstrate reasonable floor segmentation quality on the
held-out validation data.

---

## 6. ONNX Export

The trained PyTorch model was exported to ONNX.

Output:

    weights/fast_scnn_floor.onnx

ONNX Runtime was used to verify inference and benchmark model latency.

---

## 7. OpenVINO Deployment

The ONNX model was converted to OpenVINO IR:

    weights/
    ├── openvino_model.xml
    └── openvino_model.bin

OpenVINO is used as the deployment runtime because it provides a unified
interface for Intel CPU and NPU deployment.

The inference code automatically checks available devices.

Device selection logic:

    NPU available
        |
        +----> NPU
        |
        No
        |
        v
       CPU

The tested machine did not expose an NPU, so CPU execution was used for the
final benchmark.

---

## 8. Video Inference

Run PyTorch video inference:

    python video_inference_pytorch.py

Run OpenVINO video inference:

    python video_inference_openvino.py

The scripts:

- Read video frames
- Resize and normalize input
- Run segmentation
- Generate floor masks
- Apply post-processing
- Generate floor overlay
- Measure latency and FPS
- Save the resulting video

---

## 9. Benchmark

### Hardware

Test system:

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Intel CPU: used for OpenVINO CPU deployment
- NPU: Not available on the test system

OpenVINO detected:

    CPU
    GPU.0
    GPU.1

The Intel GPU backend was tested but the Fast-SCNN graph could not be
compiled due to an unsupported GPU layout operation. Therefore, CPU was used
for the OpenVINO benchmark.

### End-to-End Video Results

| Runtime | Device | Preprocess | Inference | Postprocess | Total | FPS |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | RTX 5060 | 8.95 ms | 7.28 ms | 5.54 ms | 21.76 ms | 45.96 |
| OpenVINO FP32 | CPU | 8.86 ms | 18.17 ms | 11.41 ms | 38.44 ms | 26.02 |
| OpenVINO INT8 | CPU | 12.48 ms | 14.65 ms | 14.29 ms | 41.41 ms | 24.15 |

OpenVINO FP32 achieved 26.02 FPS end-to-end, corresponding to approximately
38.44 ms per frame.

This is substantially above the required 2 Hz deployment target.

### ONNX Runtime

An inference-only ONNX Runtime benchmark measured:

    Latency: 7.64 ms
    FPS: 130.85

This measurement represents model inference only and should not be directly
compared with the end-to-end video FPS measurements above.

---

## 10. INT8 Quantization

Post-training INT8 quantization was also evaluated using NNCF.

The INT8 model reduced OpenVINO model inference latency:

    FP32: 18.17 ms
    INT8: 14.65 ms

However, total end-to-end video latency increased because preprocessing and
post-processing remained significant parts of the pipeline.

Therefore, FP32 OpenVINO was retained as the preferred deployment configuration
for this implementation.

---

## 11. Known Limitations

The model can occasionally confuse visually similar regions with floor,
especially in cluttered indoor environments.

Examples include:

- Beds
- Blankets
- Furniture
- Dark regions
- Visually similar horizontal surfaces

The current model performs binary segmentation and does not explicitly model
obstacles or traversability.

The Intel GPU backend was detected but could not compile the exported graph.
No physical Intel NPU was available for testing, although automatic NPU
selection logic is implemented.

---

## 12. Future Improvements

Potential improvements include:

- Larger and more diverse floor dataset
- ADE20K / indoor scene pretraining
- Better obstacle-aware segmentation
- Temporal smoothing for video inference
- INT8 calibration optimization
- Testing on real Intel NPU hardware
- ROS2 integration for robotic navigation
- Hardware-specific OpenVINO optimization

---

## 13. Requirements

Install dependencies:

    pip install -r requirements.txt

Example requirements:

    torch
    torchvision
    opencv-python
    numpy
    Pillow
    onnx
    onnxruntime
    openvino
    nncf

---

## 14. Project Structure

    Fast-SCNN-pytorch/
    │
    ├── train.py
    ├── export_onnx.py
    ├── video_inference_pytorch.py
    ├── video_inference_openvino.py
    ├── video_inference_openvino_int8.py
    ├── test_openvino.py
    │
    ├── models/
    │   └── fast_scnn.py
    │
    ├── data_loader/
    │   ├── __init__.py
    │   ├── floor.py
    │   └── cityscapes.py
    │
    ├── utils/
    │   ├── loss.py
    │   ├── metric.py
    │   └── lr_scheduler.py
    │
    ├── datasets/
    │   └── floor/
    │
    ├── weights/
    │   ├── fast_scnn_floor_best_model.pth
    │   ├── fast_scnn_floor.onnx
    │   ├── openvino_model.xml
    │   └── openvino_model.bin
    │
    └── test_videos/
        └── input.mp4
