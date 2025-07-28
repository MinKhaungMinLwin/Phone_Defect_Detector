from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io
import base64

app = FastAPI(title="YOLOv8 Object Detection API")
# Define allowed origins
origins = ["*"] # This allows all origins


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # List of allowed origins
    allow_credentials=True,       # Allow cookies to be included in requests
    allow_methods=["*"],          # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],          # Allow all headers
)

try:
    model = YOLO("best_3.pt")  
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/", include_in_schema=False)
async def read_root():
    """
    Servers the main index.html file."""
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the detected objects with bounding boxes.
    """
    if not model:
        return {"error": "Model Note be loaded. Please check the server logs."}
    
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    detection_count = len(results[0].boxes) if results[0].masks is not None else 0

    result_plotted = results[0].plot()

    _, buffer = cv2.imencode('.jpg', result_plotted)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "detection_count": detection_count,
        "image_base64": f"data:image/jpeg;base64,{img_base64}"
    }

