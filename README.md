# 🚗 ImuSafe Accident Detection API
Accident detection and severity classification using **FastAPI**, **YOLOv8**, **CLIP**, and rule-based reasoning.

---

## 📌 Overview

ImuSafe API processes an uploaded image and determines:

- Whether a vehicular **accident** is present  
- The **severity** of the damage (`minor`, `moderate`, `severe`)  
- Detected objects and accident indicators  
- Annotated image with bounding boxes  

The backend combines:

- **Custom YOLOv8 model** for accident detection  
- **YOLOv8 medium model** for object detection  
- **CLIP (OpenAI)** for severity understanding  
- **Rule-based system** for damage inference  
- **Fusion logic** for final severity scoring  

---

## ✨ Features

- 🚘 Accident detection using `best.pt`  
- 🧠 Severity estimation using CLIP + heuristics  
- 📦 Annotated image generation  
- 📁 Static file hosting for bounding box images  
- ⚙️ CPU-only inference for easier deployment  
- ⚡ FastAPI backend with automatic Swagger docs  

---

## 📁 Project Structure

```
imusafe-api/
│── annotated/                 # Generated annotated images
│── best.pt                    # Custom YOLO accident model
│── main.py                    # FastAPI application
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/imusafe-api.git
cd imusafe-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> Recommended Python version: **3.10+**

---

## 🚀 Running the Server

Start FastAPI using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

API will be available at:

```
http://<server-ip>:8000
```

Swagger UI:

```
http://<server-ip>:8000/docs
```

---

## 📤 API Usage

### **POST /predict**

Uploads an image and returns accident detection + severity + annotated image link.

#### Example request (cURL):

```bash
curl -X POST "http://<server-ip>:8000/predict/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Example response:

```json
{
  "accident_detected": true,
  "severity": "moderate",
  "detections": [
    {
      "class_id": 0,
      "label": "accident",
      "confidence": 0.89
    }
  ],
  "annotatedImageUrl": "http://<server-ip>:8000/annotated/8f23ab1c4e.jpg"
}
```

---

## 🧠 Severity Classification Logic

Severity comes from **two classifiers** that are **fused** into a final decision.

---

### 1. **CLIP (Vision-Language Model)**

Image is compared to 15 textual descriptions of accident severity:

- 5 minor  
- 5 moderate  
- 5 severe  

CLIP outputs the highest-scoring category →  
**`minor` | `moderate` | `severe`**

---

### 2. **Rule-Based Severity Estimation**

Rules use YOLO object detections:

| Feature | Description |
|--------|-------------|
| Damage area | % of image covered by damage/debris |
| Vehicle count | Number of vehicles in scene |
| Debris presence | Broken parts, smoke, glass |
| Collision object | Tree, pole, wall, barrier |

**Rule Logic:**

- Large damage area OR multiple vehicles → **severe**  
- Moderate debris or collision → **moderate**  
- Minimal or no debris → **minor**  

---

### 3. **Fusion Logic**

```text
If CLIP label == rule-based label → use it
If rule-based == severe → severe
If rule-based == minor and CLIP == severe → moderate
Otherwise → use CLIP label
```

This ensures stable and accurate severity scoring.

---

## 🖼️ Annotated Images

Annotated images are saved automatically in:

```
/home/ubuntu/imusafe-api/annotated/
```

Served publicly at:

```
http://<server-ip>:8000/annotated/<filename>.jpg
```

---

## 🌐 Deployment Notes

- Model inference runs on **CPU only**.
- Update this inside `main.py`:

```python
SERVER_URL = "http://<server-ip>:8000"
```

- Ensure `annotated/` is writable:

```bash
sudo chmod -R 777 annotated
```
