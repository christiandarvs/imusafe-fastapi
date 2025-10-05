from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles 
from ultralytics import YOLO
import os
import tempfile
import uuid
import shutil

app = FastAPI()

# Load YOLO model once
model = YOLO("best.pt")

# Confidence threshold
CONF_THRESHOLD = 0.7

# Directory for annotated images
ANNOTATED_DIR = "annotated"
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Serve annotated images
app.mount("/annotated", StaticFiles(directory=ANNOTATED_DIR), name="annotated")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as tmp:
        tmp.write(await file.read())
        temp_file = tmp.name

    # Run YOLO prediction
    results = model.predict(temp_file, conf=CONF_THRESHOLD, save=True)

    accident_detected = False
    detections = []

    # Prepare annotated filename
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    annotated_path = os.path.join(ANNOTATED_DIR, unique_filename)

    for r in results:
        # Collect detections
        if len(r.boxes) > 0:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                confidence = float(conf.item())
                class_id = int(cls.item())
                if confidence >= CONF_THRESHOLD:
                    detections.append({
                        "class_id": class_id,
                        "label": model.names[class_id] if hasattr(model, "names") else str(class_id),
                        "confidence": confidence,
                    })
                    if class_id == 0:  # accident class
                        accident_detected = True

        # Save annotated image
        saved_image = r.plot()  # returns numpy array
        import cv2
        cv2.imwrite(annotated_path, saved_image)

    # Clean up temp file
    os.remove(temp_file)

    SERVER_URL = "http://52.64.112.148:8000/"
    return {
        "accident_detected": accident_detected,
        "detections": detections,
        "annotatedImageUrl": f"{SERVER_URL}/annotated/{unique_filename}"
    }
