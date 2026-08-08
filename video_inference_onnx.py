import cv2
import numpy as np
import onnxruntime as ort
import time

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "C:/floor Data/Fast-SCNN-pytorch/weights/fast_scnn_floor.onnx"
INPUT_VIDEO = r"C:\floor Data\Fast-SCNN-pytorch\test_videos\input_3.mp4"
OUTPUT_VIDEO = r"C:\floor Data\Fast-SCNN-pytorch\output_floor_onnx_3.mp4"

# ==========================================================
# Load ONNX Model
# ==========================================================

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 8
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

print("✓ ONNX Model Loaded")
print("Providers:", session.get_providers())

input_name = session.get_inputs()[0].name

# ==========================================================
# Video
# ==========================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("❌ Cannot open video:", INPUT_VIDEO)
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    video_fps,
    (width, height)
)

frame_count = 0
total_time = 0

# ==========================================================
# Processing Loop
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(rgb, (512, 512))

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    # -----------------------
    # Inference
    # -----------------------

    start = time.perf_counter()

    output = session.run(None, {input_name: img})

    end = time.perf_counter()

    inference_time = end - start

    total_time += inference_time

    fps = 1 / inference_time

    # -----------------------
    # Prediction
    # -----------------------

    pred = np.argmax(output[0], axis=1)[0].astype(np.uint8)

    pred = cv2.resize(
        pred,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # =====================================================
    # Morphological Cleanup
    # =====================================================

    kernel = np.ones((5,5), np.uint8)

    pred = cv2.morphologyEx(
        pred,
        cv2.MORPH_OPEN,
        kernel
    )

    pred = cv2.morphologyEx(
        pred,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =====================================================
    # Largest Connected Component
    # =====================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred)

    mask = np.zeros_like(pred)

    if num_labels > 1:

        largest = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]

        for i in range(2, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area > largest_area:

                largest = i
                largest_area = area

        mask[labels == largest] = 1

    else:

        mask = pred

    # -----------------------
    # Overlay
    # -----------------------

    color = np.zeros_like(frame)

    color[:, :, 1] = mask * 255

    overlay = cv2.addWeighted(
        frame,
        0.75,
        color,
        0.25,
        0
    )

    # -----------------------
    # Display Info
    # -----------------------

    cv2.putText(
        overlay,
        f"FPS: {fps:.1f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        overlay,
        f"Frame: {frame_count}/{total_frames}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    out.write(overlay)

    if frame_count % 30 == 0:

        print(
            f"{frame_count}/{total_frames} | FPS: {fps:.2f}"
        )

# ==========================================================
# Finish
# ==========================================================

cap.release()
out.release()

avg_time = total_time / frame_count

print("\n====================================")
print("Video Completed")
print("====================================")
print(f"Frames       : {frame_count}")
print(f"Latency      : {avg_time*1000:.2f} ms")
print(f"Average FPS  : {1/avg_time:.2f}")
print(f"Saved        : {OUTPUT_VIDEO}")
print("====================================")