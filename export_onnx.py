import torch
from models.fast_scnn import get_fast_scnn

MODEL_PATH = "weights/fast_scnn_floor_best_model.pth"
OUTPUT_PATH = "weights/fast_scnn_floor.onnx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch model
model = get_fast_scnn(dataset="floor", aux=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✓ PyTorch model loaded")

# Dummy input
dummy_input = torch.randn(
    1, 3, 512, 512,
    device=device
)

# Export ONNX
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True
    )

print("✓ ONNX model exported successfully")
print(f"Saved: {OUTPUT_PATH}")