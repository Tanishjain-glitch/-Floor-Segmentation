import cv2
import numpy as np
import time
from openvino import Core

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = r"C:\floor Data\Fast-SCNN-pytorch\weights\openvino_model.xml"
INPUT_VIDEO = r"C:\floor Data\Fast-SCNN-pytorch\test_videos\input_3.mp4"
OUTPUT_VIDEO = r"C:\floor Data\Fast-SCNN-pytorch\output_openvino_3.mp4"

INPUT_SIZE = 512

# ==========================================================
# Load OpenVINO Model
# ==========================================================

core = Core()

devices = core.available_devices
print("Available Devices:", devices)

if "NPU" in devices:
    device = "NPU"
else:
    device = "CPU"

print("Using Device:", device)

compiled_model = core.compile_model(MODEL_PATH, device)

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print("✓ Model Loaded Successfully")

# ==========================================================
# Video
# ==========================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

fps_video = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (width, height)
)

# ==========================================================
# Normalization
# ==========================================================

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

kernel = np.ones((5, 5), np.uint8)

frame_count = 0

total_pre = 0
total_inf = 0
total_post = 0

# ==========================================================
# Processing Loop
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    ####################################################
    # Pre-processing
    ####################################################

    t0 = time.perf_counter()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))

    img = img.astype(np.float32) / 255.0

    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    t1 = time.perf_counter()

    ####################################################
    # Inference
    ####################################################

    result = compiled_model([img])[output_layer]

    t2 = time.perf_counter()

    ####################################################
    # Post-processing
    ####################################################

    mask = np.argmax(result, axis=1)[0].astype(np.uint8)

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # Morphology

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Largest Component

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    if num_labels > 1:

        largest = 1

        largest_area = stats[1, cv2.CC_STAT_AREA]

        for i in range(2, num_labels):

            if stats[i, cv2.CC_STAT_AREA] > largest_area:

                largest = i
                largest_area = stats[i, cv2.CC_STAT_AREA]

        clean_mask[labels == largest] = 1

    else:

        clean_mask = mask

    # Overlay

    overlay = frame.copy()

    green = np.zeros_like(frame)

    green[:, :, 1] = clean_mask * 255

    overlay = cv2.addWeighted(
        overlay,
        0.75,
        green,
        0.25,
        0
    )

    # Boundary

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 0, 255),
        2
    )

    t3 = time.perf_counter()

    ####################################################
    # Timing
    ####################################################

    pre = t1 - t0
    inf = t2 - t1
    post = t3 - t2

    total_pre += pre
    total_inf += inf
    total_post += post

    fps = 1 / (pre + inf + post)

    cv2.putText(
        overlay,
        f"FPS : {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        overlay,
        f"Frame : {frame_count}/{total_frames}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    writer.write(overlay)

    if frame_count % 30 == 0:

        print(
            f"{frame_count}/{total_frames} | FPS {fps:.2f}"
        )

# ==========================================================
# Finish
# ==========================================================

cap.release()
writer.release()

avg_pre = total_pre / frame_count
avg_inf = total_inf / frame_count
avg_post = total_post / frame_count

total_latency = avg_pre + avg_inf + avg_post

print("\n==========================================")

print("OpenVINO Benchmark")

print("==========================================")

print(f"Frames           : {frame_count}")

print(f"Preprocess       : {avg_pre*1000:.2f} ms")

print(f"Inference        : {avg_inf*1000:.2f} ms")

print(f"Postprocess      : {avg_post*1000:.2f} ms")

print(f"Total Latency    : {total_latency*1000:.2f} ms")

print(f"Average FPS      : {1/total_latency:.2f}")

print(f"Saved Video      : {OUTPUT_VIDEO}")

print("==========================================")