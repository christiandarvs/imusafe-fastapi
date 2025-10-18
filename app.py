from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import uuid
import os

# ====================================================
# 0. FastAPI Setup
# ====================================================
app = FastAPI()

# ====================================================
# 1. Device setup (force CPU)
# ====================================================
device = "cpu"
torch.set_default_device("cpu")
print(f"🚀 Using device: {device}")

# ====================================================
# 2. Load models
# ====================================================
print("🔹 Loading models...")
accident_model = YOLO("best.pt")  # Custom accident detection
object_model = YOLO("yolov8m.pt")  # General object detection
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ Models loaded successfully.")

# ====================================================
# 3. Directories
# ====================================================
CONF_THRESHOLD = 0.7
ANNOTATED_DIR = "annotated"
os.makedirs(ANNOTATED_DIR, exist_ok=True)
app.mount("/annotated", StaticFiles(directory=ANNOTATED_DIR), name="annotated")

# ====================================================
# 4. Helper functions
# ====================================================
def preprocess_image_for_clip(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    crop = image_rgb[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
    resized = cv2.resize(crop, (224, 224))
    return Image.fromarray(resized)


def clip_severity_label(image):
    texts = [
        # Minor
        "a car with small scratches or dents but fully functional",
        "a vehicle with minor surface damage",
        "a car showing light bumper scratches",
        "a slightly dented vehicle still drivable",
        "a car with limited visible damage",
        # Moderate
        "a car with visible damage on multiple panels",
        "a vehicle with broken headlights or cracked bumpers",
        "a moderately damaged vehicle that needs repair",
        "a car with noticeable body damage but still safe",
        "a crash with multiple damaged parts but repairable",
        # Severe
        "a severely damaged car with major structural deformation",
        "a crashed vehicle with heavy front or side damage",
        "a car with crushed panels or frame damage",
        "a heavily wrecked vehicle with broken components",
        "a total loss accident with catastrophic damage",
    ]

    inputs = clip_processor(
        text=texts,
        images=preprocess_image_for_clip(image),
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))

    if idx < 5:
        return "minor"
    elif idx < 10:
        return "moderate"
    else:
        return "severe"


def compute_damage_area(box_xywh, image_area):
    if isinstance(box_xywh, torch.Tensor):
        box_xywh = box_xywh.cpu().numpy()
    _, _, w, h = box_xywh
    return float((w * h) / image_area)


def extract_features(image):
    results = object_model(image)
    num_vehicles, has_debris, damage_area, collision_object = 0, False, 0.0, False
    h, w, _ = image.shape
    image_area = h * w

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = object_model.names[cls].lower()
            if conf < 0.5:
                continue

            if any(v in label for v in ["car", "truck", "bus", "motorcycle", "vehicle"]):
                num_vehicles += 1
            elif any(k in label for k in ["debris", "smoke", "fire", "broken", "tire", "glass", "scratch", "dent"]):
                has_debris = True
                damage_area += compute_damage_area(box.xywh[0], image_area)
            elif any(k in label for k in ["tree", "pole", "post", "wall", "barrier", "rock"]):
                collision_object = True

    return num_vehicles, damage_area, has_debris, collision_object


def fuse_severity_labels(clip_label, rule_label):
    if clip_label == rule_label:
        return clip_label
    if rule_label == "severe":
        return "severe"
    if rule_label == "minor" and clip_label == "severe":
        return "moderate"
    return clip_label


# ====================================================
# 5. API Endpoint
# ====================================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as tmp:
        tmp.write(await file.read())
        temp_file = tmp.name

    img = cv2.imread(temp_file)
    if img is None:
        os.remove(temp_file)
        return {"error": "Invalid image"}

    # YOLO accident detection
    results = accident_model.predict(temp_file, conf=CONF_THRESHOLD, save=True, device="cpu")

    accident_detected = False
    detections = []
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    annotated_path = os.path.join(ANNOTATED_DIR, unique_filename)

    for r in results:
        if len(r.boxes) > 0:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                confidence = float(conf.item())
                class_id = int(cls.item())
                if confidence >= CONF_THRESHOLD:
                    detections.append({
                        "class_id": class_id,
                        "label": accident_model.names[class_id] if hasattr(accident_model, "names") else str(class_id),
                        "confidence": confidence,
                    })
                    if class_id == 0:
                        accident_detected = True

        annotated_image = r.plot()
        cv2.imwrite(annotated_path, annotated_image)

    # Damage severity estimation
    num_vehicles, damage_area, has_debris, collision_object = extract_features(img)
    clip_label = clip_severity_label(img)

    if damage_area > 0.45 or num_vehicles >= 3 or (collision_object and damage_area > 0.25):
        rule_label = "severe"
    elif damage_area > 0.10 or has_debris or collision_object:
        rule_label = "moderate"
    else:
        rule_label = "minor"

    final_severity = fuse_severity_labels(clip_label, rule_label)

    os.remove(temp_file)

    SERVER_URL = "http://52.64.112.148:8000"
    return {
        "accident_detected": accident_detected,
        "severity": final_severity,
        "detections": detections,
        "annotatedImageUrl": f"{SERVER_URL}/annotated/{unique_filename}",
    }
