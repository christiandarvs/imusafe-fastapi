from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_file = tmp.name

    # Run YOLO prediction
    results = model.predict(temp_file, conf=CONF_THRESHOLD, save=False)

    accident_detected = False
    detections = []

    # Annotated file path
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    annotated_path = os.path.join(ANNOTATED_DIR, unique_filename)

    for r in results:
        if len(r.boxes) > 0:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                class_id = int(cls.item())
                confidence = float(conf.item())

                # Only keep detections above threshold
                if confidence >= CONF_THRESHOLD:
                    detections.append({
                        "class_id": class_id,
                        "label": model.names[class_id] if hasattr(model, "names") else str(class_id),
                        "confidence": confidence,
                    })

                    if class_id == 0:  # accident class
                        accident_detected = True

        # Save annotated image
        r.save(filename=annotated_path)

    # Clean up temp file
    os.remove(temp_file)

    return {
        "accident_detected": accident_detected,
        "detections": detections,
        "annotated_image_url": f"/annotated/{unique_filename}"
    }

# Serve annotated images
@app.get("/annotated/{filename}")
async def get_annotated_image(filename: str):
    filepath = os.path.join(ANNOTATED_DIR, filename)
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    return FileResponse(filepath)
