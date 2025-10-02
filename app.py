from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO inference
    results = model.predict(temp_file, imgsz=640, conf=0.5)

    # Clean up
    os.remove(temp_file)

    # Extract labels
    accident_detected = any(r.boxes.cls[0].item() == 0 for r in results)  # adjust based on your class index
    
    return {"accident_detected": bool(accident_detected)}
