from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import os
import tempfile

app = FastAPI()

# Load YOLO model once at startup
model = YOLO("best.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Create a temporary file in /tmp (safe and always writable)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_file = tmp.name

    # Run YOLO prediction
    results = model.predict(temp_file)

    accident_detected = False
    for r in results:
        if len(r.boxes) > 0:  # ✅ check if detections exist
            if any(cls.item() == 0 for cls in r.boxes.cls):
                accident_detected = True

    # Clean up temp file
    os.remove(temp_file)

    return {"accident_detected": accident_detected}
