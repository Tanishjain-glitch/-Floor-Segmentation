import cv2
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms

from models.fast_scnn import get_fast_scnn

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = r"C:\floor Data\Fast-SCNN-pytorch\weights\fast_scnn_floor_best_model.pth"

INPUT_VIDEO = r"C:\floor Data\Fast-SCNN-pytorch\test_videos\input_3.mp4"

OUTPUT_VIDEO = r"C:\floor Data\Fast-SCNN-pytorch\output_pytorch_3.mp4"

INPUT_SIZE = 512

DEVICE =  "cpu"

print("Using Device:", DEVICE)

# ==========================================================
# Load Model
# ==========================================================

model = get_fast_scnn(
    dataset="floor",
    aux=False
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.to(DEVICE)
model.eval()

print("✓ Model Loaded")

# ==========================================================
# Transform
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# Open Video
# ==========================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps_video = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (width,height)
)

kernel = np.ones((5,5), np.uint8)

frame_count = 0

total_pre = 0
total_inf = 0
total_post = 0

print("================================")
print("Starting Inference...")
print("================================")

# ==========================================================
# Loop
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    #######################################################
    # Preprocessing
    #######################################################

    t0 = time.perf_counter()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    t1 = time.perf_counter()

    #######################################################
    # Inference
    #######################################################

    with torch.no_grad():

        output = model(tensor)

        pred = torch.argmax(
            output[0],
            dim=1
        )

    t2 = time.perf_counter()

    #######################################################
    # Postprocessing
    #######################################################

    mask = pred.squeeze().cpu().numpy().astype(np.uint8)

    mask = cv2.resize(
        mask,
        (width,height),
        interpolation=cv2.INTER_NEAREST
    )

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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    if num_labels > 1:

        largest = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]

        for i in range(2, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area > largest_area:

                largest = i
                largest_area = area

        clean_mask[labels == largest] = 1

    else:

        clean_mask = mask

    #######################################################
    # Overlay
    #######################################################

    green = np.zeros_like(frame)

    green[:,:,1] = clean_mask * 255

    overlay = cv2.addWeighted(
        frame,
        0.75,
        green,
        0.25,
        0
    )

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0,0,255),
        2
    )

    t3 = time.perf_counter()

    #######################################################
    # Timing
    #######################################################

    pre = t1 - t0
    inf = t2 - t1
    post = t3 - t2

    total_pre += pre
    total_inf += inf
    total_post += post

    total = pre + inf + post

    fps = 1 / total

    cv2.putText(
        overlay,
        f"FPS : {fps:.1f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        overlay,
        f"Frame : {frame_count}/{total_frames}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
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

latency = avg_pre + avg_inf + avg_post

print("\n==========================================")

print("PyTorch Benchmark")

print("==========================================")

print(f"Frames           : {frame_count}")

print(f"Preprocess       : {avg_pre*1000:.2f} ms")

print(f"Inference        : {avg_inf*1000:.2f} ms")

print(f"Postprocess      : {avg_post*1000:.2f} ms")

print(f"Total Latency    : {latency*1000:.2f} ms")

print(f"Average FPS      : {1/latency:.2f}")

print(f"Saved Video      : {OUTPUT_VIDEO}")

print("==========================================")