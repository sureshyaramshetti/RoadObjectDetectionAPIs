import uuid
import shutil
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
import mimetypes
import cv2

app = FastAPI()
model = YOLO("../model/best.pt")  # Your model here

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Save input image
    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    output_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run detection
    results = model.predict(input_path, save=True, project=OUTPUT_DIR, name=image_id)
    saved_path = os.path.join(OUTPUT_DIR, image_id, os.path.basename(input_path))

    return FileResponse(saved_path, media_type="image/jpeg", filename="predicted.jpg")

@app.post("/track/video")
async def track_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[-1]  # get uploaded file extension (e.g., .mov)
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run tracking
    results = model.track(source=input_path, save=True, save_txt=False, save_conf=True, project=OUTPUT_DIR, name=video_id)
    saved_path = os.path.join(OUTPUT_DIR, video_id, f"{video_id}.avi")

    # Determine MIME type from file extension
    mime_type, _ = mimetypes.guess_type(saved_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Fallback if type is unknown

    return FileResponse(saved_path, media_type=mime_type, filename=f"tracked_video.avi")

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[-1]  # get uploaded file extension (e.g., .mov)
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file,f)

    model = YOLO("model/best.pt")

    #run prediction
    results = model.predict(source= input_path, save=True, project=OUTPUT_DIR, name=video_id, save_txt=False, save_conf=True)
    saved_path = os.path.join(OUTPUT_DIR, video_id, f"{video_id}.avi")

    # Determine MIME type from file extension
    mime_type, _ = mimetypes.guess_type(saved_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Fallback if type is unknown

    return FileResponse(saved_path, media_type=mime_type, filename=f"predicted.avi")