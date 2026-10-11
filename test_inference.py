import glob
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torchvision.transforms as transforms

# Define target size expected by MobileNetV3 / standard vision models
TARGET_SIZE = (224, 224)


def letterbox_preprocess(image_path, target_size=TARGET_SIZE):
  """Pads image to square without distorting aspect ratio, resizes, and normalizes."""
  img = Image.open(image_path).convert("RGB")
  w, h = img.size
  max_dim = max(w, h)

  # 1. Create a black square canvas
  padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
  # 2. Center the original image on the canvas
  padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

  # 3. Resize to expected target dimensions
  resized = padded.resize(target_size, Image.Resampling.BILINEAR)

  # 4. Normalize (ImageNet standard: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  img_np = np.array(resized, dtype=np.float32) / 255.0
  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  normalized = (img_np - mean) / std

  # 5. Convert HWC -> CHW and add batch dimension -> (1, C, H, W)
  tensor = np.transpose(normalized, (2, 0, 1))
  tensor_batch = np.expand_dims(tensor, axis=0).astype(np.float32)
  return tensor_batch


def test_onnx_model(onnx_path, image_paths):
  """Runs inference on .onnx model using ONNX Runtime (CPU)."""
  print(f"\n==========================================")
  print(f"Testing ONNX Model: {onnx_path}")
  print(f"==========================================")

  if not os.path.exists(onnx_path):
    print(f"Error: {onnx_path} not found.")
    return

  session = ort.InferenceSession(onnx_path)
  input_name = session.get_inputs()[0].name

  for img_p in image_paths:
    input_data = letterbox_preprocess(img_p)
    outputs = session.run(None, {input_name: input_data})
    raw_output = outputs[0]

    # Compute softmax probabilities
    exp_out = np.exp(raw_output - np.max(raw_output))
    probs = exp_out / np.sum(exp_out)
    pred_class = np.argmax(probs)

    print(
        f"[{os.path.basename(img_p)}] -> Predicted Class: {pred_class} | Raw"
        f" Logits: {raw_output.flatten()}"
    )


def test_pth_model(pth_path, image_paths):
  """Runs inference on PyTorch .pth model."""
  print(f"\n==========================================")
  print(f"Testing PyTorch Model: {pth_path}")
  print(f"==========================================")

  if not os.path.exists(pth_path):
    print(f"Error: {pth_path} not found.")
    return

  try:
    checkpoint = torch.load(pth_path, map_location=torch.device("cpu"))

    # Check if loaded object is a full model or state_dict
    if isinstance(checkpoint, torch.nn.Module):
      model = checkpoint
    else:
      print(
        f"Warning: {pth_path} contains a state_dict. Make sure to instantiate"
        " the model architecture before loading weights."
      )
      return

    model.eval()

    with torch.no_grad():
      for img_p in image_paths:
        input_data = letterbox_preprocess(img_p)
        input_tensor = torch.from_numpy(input_data)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()
        pred_class = np.argmax(probs)

        print(
            f"[{os.path.basename(img_p)}] -> Predicted Class: {pred_class} |"
            f" Probs: {probs.flatten()}"
        )

  except Exception as e:
    print(f"Failed to run {pth_path}: {e}")


if __name__ == "__main__":
  # Find all coffee bean test images in test/ folder
  test_images = sorted(
      glob.glob("test/coffee-bean-*.jpeg") + glob.glob("test/coffee-bean-*.jpg")
  )

  if not test_images:
    print("No images matching 'test/coffee-bean-<index>.jpeg' found!")
  else:
    print(f"Found {len(test_images)} test image(s):")
    for img in test_images:
      print(f" - {img}")

    # List of models in your directory
    onnx_models = ["coffee_mobilenetv3_hailo.onnx"]
    pth_models = [
        "stage1_ds1_github.pth",
        "stage1_ds2_kagglev1.pth",
        "stage2_fused_final.pth",
    ]

    # Test ONNX Model
    for model_file in onnx_models:
      test_onnx_model(model_file, test_images)

    # Test PyTorch Models
    for model_file in pth_models:
      test_pth_model(model_file, test_images)
