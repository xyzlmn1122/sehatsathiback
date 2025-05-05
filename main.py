<<<<<<< HEAD
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import tempfile
import shutil
import os
import io
import base64

app = FastAPI()

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZFjSQlPU6iLZKMbfc00P"
)

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    # Read and process image
    image = cv2.imread(temp_path)
    height, width, _ = image.shape
    result = CLIENT.infer(temp_path, model_id="skin-3n2jd-7vrcl/1")
    os.remove(temp_path)  # Clean up

    predictions = result.get("predictions", [])

    if not predictions:
        return JSONResponse(content={
            "is_healthy": True,
            "message": "No skin condition detected.",
            "predictions": [],
            "annotated_image": None
        })

    issues = []
    
    # Annotate the image with bounding boxes and labels
    for pred in predictions:
        # Get bounding box coordinates
        x1 = int(pred.get('x', 0))
        y1 = int(pred.get('y', 0))
        x2 = int(pred.get('x2', 0))
        y2 = int(pred.get('y2', 0))

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, f"{pred['class']} ({pred['confidence']*100:.2f}%)", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Add issue details to response
        issues.append({
            "label": pred["class"],
            "confidence": pred["confidence"],
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    # Convert annotated image to base64
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={
        "is_healthy": False,
        "message": "Possible skin condition detected.",
        "predictions": issues,
        "annotated_image": image_base64
    })
=======
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import tempfile
import shutil
import os
import io
import base64

app = FastAPI()

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZFjSQlPU6iLZKMbfc00P"
)

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    # Read and process image
    image = cv2.imread(temp_path)
    height, width, _ = image.shape
    result = CLIENT.infer(temp_path, model_id="skin-3n2jd-7vrcl/1")
    os.remove(temp_path)  # Clean up

    predictions = result.get("predictions", [])

    if not predictions:
        return JSONResponse(content={
            "is_healthy": True,
            "message": "No skin condition detected.",
            "predictions": [],
            "annotated_image": None
        })

    issues = []
    
    # Annotate the image with bounding boxes and labels
    for pred in predictions:
        # Get bounding box coordinates
        x1 = int(pred.get('x', 0))
        y1 = int(pred.get('y', 0))
        x2 = int(pred.get('x2', 0))
        y2 = int(pred.get('y2', 0))

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, f"{pred['class']} ({pred['confidence']*100:.2f}%)", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Add issue details to response
        issues.append({
            "label": pred["class"],
            "confidence": pred["confidence"],
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    # Convert annotated image to base64
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={
        "is_healthy": False,
        "message": "Possible skin condition detected.",
        "predictions": issues,
        "annotated_image": image_base64
    })
>>>>>>> 1b14179 (Fix vercel deployment config)
