import cv2
import nncf
import numpy as np
from pathlib import Path
from openvino import Core, save_model

# -------------------------------------------------
# Paths
# -------------------------------------------------

FP32_MODEL = "weights/openvino_model.xml"

OUTPUT_MODEL = "weights/openvino_int8.xml"

CALIB_DIR = "calibration"

INPUT_SIZE = 512

# -------------------------------------------------
# Dataset
# -------------------------------------------------

image_paths = sorted(Path(CALIB_DIR).glob("*"))

print(f"Calibration Images : {len(image_paths)}")

mean = np.array([0.485,0.456,0.406],dtype=np.float32)
std = np.array([0.229,0.224,0.225],dtype=np.float32)


def transform(path):

    img = cv2.imread(str(path))

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = cv2.resize(img,(INPUT_SIZE,INPUT_SIZE))

    img = img.astype(np.float32)/255.0

    img = (img-mean)/std

    img = np.transpose(img,(2,0,1))

    img = np.expand_dims(img,0)

    return img


dataset = nncf.Dataset(
    image_paths,
    transform
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

core = Core()

model = core.read_model(FP32_MODEL)

print("FP32 Model Loaded")

# -------------------------------------------------
# Quantization
# -------------------------------------------------

quantized_model = nncf.quantize(
    model,
    dataset
)

print("Quantization Finished")

# -------------------------------------------------
# Save
# -------------------------------------------------

save_model(
    quantized_model,
    OUTPUT_MODEL
)

print("\nINT8 Model Saved")

print(OUTPUT_MODEL)