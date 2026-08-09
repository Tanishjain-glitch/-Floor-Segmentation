# Fast-SCNN Floor Segmentation for Edge Deployment

Real-time semantic floor segmentation using **Fast-SCNN**, with deployment
through **PyTorch, ONNX Runtime, and Intel OpenVINO**.

The system takes an RGB image or video stream and produces a segmentation mask
identifying the **floor region**.

<p align="center">
  <img src="assets/demo.gif" alt="Floor segmentation demo" width="700">
</p>

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

## 3. Dataset

A custom indoor floor segmentation dataset was used for training and
evaluation.

Only the **Floor** region is manually annotated. All remaining pixels
represent non-floor/background regions.

### Dataset Split

| Split      |    Images |
| ---------- | --------: |
| Training   | **7,128** |
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
## Demo

<p align="center">
  <img src="demo/FLOOR_GIF.gif" width="350">
</p>

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
Latency            : 11.93 ms
Throughput         : 83.86 FPS
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

Run OpenVINO video inference (FP32):

```bash
python video_inference_openvino.py
```

Run OpenVINO video inference (INT8, **final deployment configuration**):

```bash
python video_inference_openvino_int8.py
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
| **PyTorch**       | RTX 5060  |    8.53 ms |   7.37 ms |     5.11 ms |  **21.02 ms** | **47.58** |
| **PyTorch**       | Intel CPU |    8.12 ms |  48.05 ms |     4.72 ms |  **60.88 ms** | **16.42** |
| **OpenVINO FP32** | Intel CPU |    7.09 ms |  12.70 ms |     8.90 ms |  **28.68 ms** | **34.86** |
| **OpenVINO INT8** | Intel CPU |    6.78 ms |   9.21 ms |     8.21 ms |  **24.20 ms** | **41.33** |

### Performance Target

Required deployment target:

```text
Minimum Rate    : 2 Hz
Maximum Latency : 500 ms/frame
```

Final OpenVINO INT8 result:

```text
Latency : 24.20 ms/frame
FPS     : 41.33
Hz      : 41.33
```

The OpenVINO INT8 implementation therefore comfortably exceeds the required
real-time deployment target.

---

## 11. INT8 Quantization

Post-training INT8 quantization was evaluated using **NNCF**.

### Inference Latency

```text
FP32 : 12.70 ms
INT8 : 9.21 ms
```

INT8 reduced the model inference latency by approximately **27.5%**.

The complete video pipeline improved as well:

```text
FP32 : 28.68 ms/frame
INT8 : 24.20 ms/frame
```

This is roughly a **15.6% reduction in end-to-end latency**, closely tracking
the inference-side speedup rather than being offset by fixed preprocessing/
postprocessing overhead.

Therefore, **OpenVINO INT8 was selected as the final deployment
configuration**, offering the lowest latency and highest throughput of any
configuration tested.

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
│   └── openvino_model.bin
│
└── test_videos/
    └── input.mp4
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

* PyTorch CUDA: **47.58 FPS / 21.02 ms**
* PyTorch CPU: **16.42 FPS / 60.88 ms**
* ONNX Runtime: **83.86 FPS / 11.93 ms**
* OpenVINO FP32: **34.86 FPS / 28.68 ms**
* OpenVINO INT8: **41.33 FPS / 24.20 ms**

The **OpenVINO INT8 configuration** was selected as the final Intel-oriented
deployment because it delivers the lowest latency and highest throughput of
any configuration tested, comfortably exceeding the required **2 Hz
real-time target** while maintaining the validated floor segmentation
performance.
