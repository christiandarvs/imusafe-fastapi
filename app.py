from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import os
import tempfile

app = FastAPI()

# Load YOLO model once
model = YOLO("best.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_file = tmp.name

    results = model.predict(temp_file)

    accident_detected = False
    detections = []

    for r in results:
        if len(r.boxes) > 0:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                class_id = int(cls.item())
                confidence = float(conf.item())

                detections.append({
                    "class_id": class_id,
                    "label": model.names[class_id] if hasattr(model, "names") else str(class_id),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

                if class_id == 0:  # accident class
                    accident_detected = True

    os.remove(temp_file)

    return {
        "accident_detected": accident_detected,
        "detections": detections
    }
