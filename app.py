from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    contents = await file.read()
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(contents)

    # Run YOLO prediction
    results = model.predict(temp_file)

    accident_detected = False
    for r in results:
        if len(r.boxes) > 0:  # ✅ only check if detections exist
            if any(cls.item() == 0 for cls in r.boxes.cls):
                accident_detected = True

    os.remove(temp_file)

    return {"accident_detected": accident_detected}

