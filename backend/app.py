# ================================================================
# app.py — Plant Disease Detection API
# ================================================================
# Model Analysis Results:
#   Architecture : MobileNetV2 (pretrained, fine-tuned)
#   Input        : RGB image, resized to 224×224
#   Output       : 38 plant disease classes (PlantVillage dataset)
#   Classifier   : Linear(1280 → 38)
# ================================================================

import os
import io
import base64
import json
from datetime import datetime
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App Setup ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Class Labels (PlantVillage 38-class dataset) ───────────────
CLASS_NAMES = [
    "Apple — Apple Scab",
    "Apple — Black Rot",
    "Apple — Cedar Apple Rust",
    "Apple — Healthy",
    "Blueberry — Healthy",
    "Cherry — Powdery Mildew",
    "Cherry — Healthy",
    "Corn — Cercospora / Gray Leaf Spot",
    "Corn — Common Rust",
    "Corn — Northern Leaf Blight",
    "Corn — Healthy",
    "Grape — Black Rot",
    "Grape — Esca (Black Measles)",
    "Grape — Leaf Blight",
    "Grape — Healthy",
    "Orange — Huanglongbing (Citrus Greening)",
    "Peach — Bacterial Spot",
    "Peach — Healthy",
    "Pepper Bell — Bacterial Spot",
    "Pepper Bell — Healthy",
    "Potato — Early Blight",
    "Potato — Late Blight",
    "Potato — Healthy",
    "Raspberry — Healthy",
    "Soybean — Healthy",
    "Squash — Powdery Mildew",
    "Strawberry — Leaf Scorch",
    "Strawberry — Healthy",
    "Tomato — Bacterial Spot",
    "Tomato — Early Blight",
    "Tomato — Late Blight",
    "Tomato — Leaf Mold",
    "Tomato — Septoria Leaf Spot",
    "Tomato — Spider Mites",
    "Tomato — Target Spot",
    "Tomato — Yellow Leaf Curl Virus",
    "Tomato — Mosaic Virus",
    "Tomato — Healthy",
]

# Whether the prediction is a disease or a healthy plant
IS_HEALTHY = [
    False, False, False, True,
    True,
    False, True,
    False, False, False, True,
    False, False, False, True,
    False,
    False, True,
    False, True,
    False, False, True,
    True, True, False,
    False, True,
    False, False, False, False, False, False, False, False, False, True,
]

# ── Image Preprocessing ────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Load MobileNetV2 Model ─────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_plant_disease_model.pth')

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 38)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("✅ Plant disease model loaded successfully!")
    print(f"   Path : {MODEL_PATH}")
    print(f"   Classes: {len(CLASS_NAMES)}")
except FileNotFoundError:
    print(f"⚠️ Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

# ── JSON Logging Function ──────────────────────────────────────
def save_prediction_to_json(data):
    file_path = "prediction_logs.json"

    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logs.append(data)

        with open(file_path, "w") as f:
            json.dump(logs, f, indent=4)

    except Exception as e:
        print("Error saving prediction:", e)


def _jet_colormap(cam):
    """Map a [0,1] CAM array to an RGB heatmap using a Jet-like palette."""
    four = 4.0 * cam
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def generate_gradcam_overlay(pil_img, input_tensor, target_class_idx):
    """Generate Grad-CAM overlay for MobileNetV2 and return base64 PNG string."""
    try:
        target_layer = model.features[-1][0]
        activations = []
        gradients = []

        def _forward_hook(_module, _inputs, output):
            activations.append(output.detach())

        def _backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        fwd_handle = target_layer.register_forward_hook(_forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

        was_training = model.training
        model.eval()
        
        # Enable gradients for input
        input_tensor_req = input_tensor.clone().detach().requires_grad_(True)
        
        try:
            # Forward pass
            output = model(input_tensor_req)
            class_score = output[:, target_class_idx]
            
            # Backward pass
            model.zero_grad(set_to_none=True)
            class_score.backward()

            if not activations or not gradients:
                print("⚠️  Grad-CAM: No activations or gradients captured")
                return None

            grad = gradients[0]
            act = activations[0]

            # Compute weights
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # Resize to input image size
            cam = F.interpolate(
                cam,
                size=(pil_img.height, pil_img.width),
                mode='bilinear',
                align_corners=False,
            )

            # Normalize CAM
            cam_min, cam_max = cam.min(), cam.max()
            if (cam_max - cam_min).item() < 1e-8:
                print("⚠️  Grad-CAM: CAM range too small")
                return None

            cam = (cam - cam_min) / (cam_max - cam_min)
            cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)

            # Generate heatmap
            heatmap = _jet_colormap(cam_np)
            original = np.asarray(pil_img).astype(np.float32) / 255.0
            overlay = np.clip(0.55 * original + 0.45 * heatmap, 0.0, 1.0)
            overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

            # Encode to base64
            buffer = io.BytesIO()
            overlay_img.save(buffer, format='PNG')
            result = base64.b64encode(buffer.getvalue()).decode('utf-8')
            print("✅ Grad-CAM generated successfully")
            return result
        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            if was_training:
                model.train()
    except Exception as e:
        print(f"⚠️  Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ── Routes ─────────────────────────────────────────────────────

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "model": "MobileNetV2 — Plant Disease Detector",
        "classes": len(CLASS_NAMES)
    })


@app.route('/predict', methods=['POST'])
def predict():
    print("FILES:", request.files)
    print("FORM:", request.form)
    print("JSON:", request.get_json(silent=True))

    try:

        # ── Read image ──────────────────────────────────────────
        if 'image' in request.files or 'file' in request.files:
            file = request.files.get('image') or request.files.get('file')
            img = Image.open(file.stream).convert('RGB')

        elif request.is_json and 'image' in request.get_json():
            data = request.get_json()
            img_bytes = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        else:
            return jsonify({"error": "No image provided."}), 400

        # ── Preprocess ─────────────────────────────────────────
        tensor = preprocess(img).unsqueeze(0)

        # ── Inference ──────────────────────────────────────────
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1).squeeze()

        # ── Top5 predictions ───────────────────────────────────
        top5_probs, top5_indices = torch.topk(probabilities, 5)

        top5 = [
            {
                "label": CLASS_NAMES[idx.item()],
                "confidence": round(prob.item(), 4),
                "is_healthy": IS_HEALTHY[idx.item()]
            }
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        best_idx = top5_indices[0].item()
        gradcam_overlay_b64 = generate_gradcam_overlay(img, tensor, best_idx)

        result = {
            "prediction": CLASS_NAMES[best_idx],
            "index": best_idx,
            "confidence": round(top5_probs[0].item(), 4),
            "is_healthy": IS_HEALTHY[best_idx],
            "top5": top5,
            "gradcam_overlay": gradcam_overlay_b64
        }

        # ── Save JSON log ──────────────────────────────────────
        log_payload = {k: v for k, v in result.items() if k != "gradcam_overlay"}
        save_prediction_to_json(log_payload)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)