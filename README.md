# Fast-SCNN Floor Segmentation for Edge Deployment

Real-time semantic floor segmentation using **Fast-SCNN**, with deployment
through **PyTorch, ONNX Runtime, and Intel OpenVINO**.

The system takes an RGB image or video stream and produces a segmentation mask
identifying the **floor region**.

---

## 1. Overview

This project implements a lightweight semantic segmentation pipeline for
indoor floor detection.

The model is trained on a custom indoor floor segmentation dataset with
**one annotated foreground class: Floor**.

### Target Class

- **Floor** — annotated foreground class
- **Background** — non-floor region

The trained Fast-SCNN model is exported to ONNX and converted to OpenVINO IR
for edge-oriented deployment.

### Deployment Targets

- PyTorch + CUDA
- ONNX Runtime
- OpenVINO CPU
- OpenVINO NPU fallback logic

---

## 2. Architecture

Fast-SCNN was selected because it is designed for efficient semantic
segmentation while maintaining a lightweight architecture suitable for
real-time applications.

The architecture consists of:

- Learning-to-Downsample module
- Global Feature Extractor
- Pyramid Pooling
- Feature Fusion

### Inference Pipeline

```text
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
```

### Deployment Pipeline

```text
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
CPU / NPU Deployment
```

---

## Demo

<p align="center">
  <img src="demo/FLOOR_GIF.gif" width="350">
</p>

## 3. Dataset

A custom indoor floor segmentation dataset was used for training and
evaluation.

Only the **Floor** region is manually annotated. All remaining pixels
represent non-floor/background regions.

### Dataset Split

| Split      |    Images |
| ---------- | --------: |
| Training   | **7,148** |
| Validation |   **681** |
| Testing    |   **681** |

### Dataset Structure

```text
datasets/
└── floor/
    ├── train/
    │   ├── images/
    │   └── masks/
    │
    ├── val/
    │   ├── images/
    │   └── masks/
    │
    └── test/
        ├── images/
        └── masks/
```

### Mask Convention

```text
0 = Background
1 = Floor
```

The dataset contains one manually annotated foreground class, **Floor**.

---

## 4. Preprocessing and Augmentation

The training dataset uses the following preprocessing and augmentation
pipeline.

### Preprocessing

* **Auto-Orient:** Applied
* **Resize:** 512 × 512

### Data Augmentation

The training data uses:

* Horizontal Flip
* Rotation: **−10° to +10°**
* Hue: **−5° to +5°**
* Saturation: **−10% to +10%**
* Brightness: **−15% to +15%**
* Exposure: **−15% to +15%**
* Blur: Up to **1.2 px**
* Noise: Up to **0.5% of pixels**
* Motion Blur: **100 px length, 0° angle**

These augmentations improve robustness to changes in camera orientation,
lighting, image quality, and motion.

---

## 5. Training

The model was trained using:

* **Model:** Fast-SCNN
* **Annotated foreground class:** Floor
* **Model output:** Background + Floor
* **Input Resolution:** 512 × 512
* **Batch Size:** 8
* **Optimizer:** SGD
* **Initial Learning Rate:** 0.001
* **Momentum:** 0.9
* **Weight Decay:** 0.0001
* **Epochs:** 100

Training command:

```bash
python train.py --dataset floor
```

The best trained model is:

```text
weights/fast_scnn_floor_best_model.pth
```

---

## 6. Validation Results

The trained model achieved approximately:

* **Pixel Accuracy:** 94.9%
* **mIoU:** 87.8%

These results were obtained on the held-out validation data.

The results demonstrate that the trained Fast-SCNN model can effectively
identify floor regions in the evaluated indoor environments.

---

## 7. ONNX Export

The trained PyTorch model was exported to ONNX.

Output:

```text
weights/fast_scnn_floor.onnx
```

ONNX Runtime was used to verify the exported model and benchmark inference.

### ONNX Runtime Benchmark

```text
Execution Provider : CPUExecutionProvider
Latency            : 9.38 ms
Throughput         : 106.63 FPS
```

> The ONNX benchmark measures the ONNX Runtime inference call and should not
> be interpreted as the complete end-to-end video pipeline latency.

---

## 8. OpenVINO Deployment

The ONNX model was converted to OpenVINO IR:

```text
weights/
├── openvino_model.xml
└── openvino_model.bin
```

OpenVINO provides a unified runtime for Intel hardware.

The implementation checks available devices and can select an NPU when one
is available, with CPU fallback.

### Tested Hardware

* **CPU:** Intel® Core™ 7 240H @ 2.50 GHz
* **RAM:** 16 GB
* **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB
* **NPU:** Not available

OpenVINO detected:

```text
CPU
GPU.0
GPU.1
```

The Intel GPU backend was tested, but the Fast-SCNN graph could not be compiled
because of an unsupported GPU layout operation.

Therefore, the final OpenVINO benchmark was performed on the **Intel CPU**.

---

## 9. Video Inference

### PyTorch

Run PyTorch video inference:

```bash
python video_inference_pytorch.py
```

### OpenVINO

Run OpenVINO video inference:

```bash
python video_inference_openvino.py
```

The inference scripts:

* Read video frames
* Resize input to 512 × 512
* Normalize input
* Run semantic segmentation
* Generate floor masks
* Apply post-processing
* Generate floor overlays
* Measure latency and FPS
* Save the output video

---

## 10. Benchmark

### End-to-End Video Benchmark

| Runtime           | Device    | Preprocess | Inference | Postprocess | Total Latency |       FPS |
| ----------------- | --------- | ---------: | --------: | ----------: | ------------: | --------: |
| **PyTorch**       | RTX 5060  |    8.95 ms |   7.28 ms |     5.54 ms |  **21.76 ms** | **45.96** |
| **PyTorch**       | Intel CPU |   16.70 ms |  94.77 ms |     8.65 ms | **120.11 ms** |  **8.33** |
| **OpenVINO FP32** | Intel CPU |    8.86 ms |  18.17 ms |    11.41 ms |  **38.44 ms** | **26.02** |
| **OpenVINO INT8** | Intel CPU |   12.48 ms |  14.65 ms |    14.29 ms |  **41.41 ms** | **24.15** |

### Performance Target

Required deployment target:

```text
Minimum Rate    : 2 Hz
Maximum Latency : 500 ms/frame
```

Final OpenVINO FP32 result:

```text
Latency : 38.44 ms/frame
FPS     : 26.02
Hz      : 26.02
```

The OpenVINO FP32 implementation therefore comfortably exceeds the required
real-time deployment target.

---

## 11. INT8 Quantization

Post-training INT8 quantization was evaluated using **NNCF**.

### Inference Latency

```text
FP32 : 18.17 ms
INT8 : 14.65 ms
```

INT8 reduced the model inference latency by approximately **19%**.

However, the complete video pipeline became slightly slower:

```text
FP32 : 38.44 ms/frame
INT8 : 41.41 ms/frame
```

The increase was mainly due to preprocessing and post-processing overhead.

Therefore, **OpenVINO FP32 was retained as the preferred deployment
configuration**.

---

## 12. Known Limitations

The model can occasionally confuse visually similar regions with floor,
particularly in cluttered indoor environments.

Observed challenging cases include:

* Beds
* Blankets
* Furniture
* Dark regions and shadows
* Visually similar horizontal surfaces
* Cluttered indoor environments

The current model performs floor/background segmentation and does not
explicitly classify different obstacle categories.

Therefore, additional obstacle detection or traversability estimation would
be required for direct robot navigation.

Performance may also decrease when the model encounters:

* Different lighting conditions
* Different floor textures
* Unseen environments
* Significantly different camera viewpoints
* Heavy scene clutter

---

## 13. Future Improvements

Potential future improvements include:

* Increasing dataset diversity
* Adding more indoor environments
* Temporal smoothing for video inference
* Obstacle-aware segmentation
* Traversability estimation
* Further OpenVINO optimization
* Hardware-specific optimization
* Testing on Intel NPU hardware
* ROS2 integration for robotic navigation

---

## 14. Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
torchvision
opencv-python
numpy
Pillow
onnx
onnxruntime
openvino
nncf
```

---

## 15. Project Structure

```text
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
│       ├── train/
│       ├── val/
│       └── test/
│
├── weights/
│   ├── fast_scnn_floor_best_model.pth
│   ├── fast_scnn_floor.onnx
│   ├── openvino_model.xml
│   |── openvino_model.bin
|   ├── openvino_int8.xml
│   └── openvino_int8.bin
│
|── test_videos/
|   └── input.mp4
└── outputs/
    └── output_openvino_3.mp4
```

---

## 16. Results Summary

The project demonstrates a complete pipeline from custom dataset training to
optimized edge deployment:

```text
Custom Floor Dataset
        ↓
Floor Annotation
        ↓
Fast-SCNN Training
        ↓
PyTorch (.pth)
        ↓
ONNX Export
        ↓
ONNX Runtime
        ↓
OpenVINO Conversion
        ↓
CPU Deployment
        ↓
Real-Time Floor Segmentation
```

### Final Results

**Segmentation Quality**

* Pixel Accuracy: **94.9%**
* mIoU: **87.8%**

**Deployment Performance**

* PyTorch CUDA: **45.96 FPS**
* PyTorch CPU: **8.33 FPS**
* ONNX Runtime: **106.63 FPS / 9.38 ms**
* OpenVINO FP32: **26.02 FPS / 38.44 ms**
* OpenVINO INT8: **24.15 FPS / 41.41 ms**

The **OpenVINO FP32 configuration** was selected as the final Intel-oriented
deployment because it exceeds the required **2 Hz real-time target** while
maintaining the validated floor segmentation performance.
