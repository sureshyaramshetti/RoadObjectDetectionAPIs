import uuid
import shutil
import os
import logging
import traceback
import mimetypes
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
from fastapi_utils.tasks import repeat_every  # Install via `pip install fastapi-utils`

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
try:
    model_path = "../model/best.pt"
    logger.info(f"Attempting to load model from {model_path}")
    model = YOLO(model_path)
    logger.info("Model loaded successfully.")
except Exception:
    logger.error("Failed to load model:")
    logger.error(traceback.format_exc())
    model = None

# === Periodic Cleanup Task ===
@app.on_event("startup")
@repeat_every(seconds=900)  # every 15 minutes
def cleanup_old_files():
    def remove_old_files_from(dir_path):
        now = time.time()
        cutoff = now - 1800  # 30 minutes = 1800 seconds
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < cutoff:
                        os.remove(file_path)
                        logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

    remove_old_files_from(UPLOAD_DIR)
    remove_old_files_from(OUTPUT_DIR)
    logger.info("Completed periodic cleanup.")

# === Predict Image Endpoint ===
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    logger.info("Received image for prediction")
    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved input image to {input_path}")

    if model is None:
        return {"error": "Model not loaded"}

    results = model.predict(input_path, save=True, project=OUTPUT_DIR, name=image_id)
    saved_path = os.path.join(OUTPUT_DIR, image_id, os.path.basename(input_path))
    logger.info(f"Prediction completed. Saved to {saved_path}")

    return FileResponse(saved_path, media_type="image/jpeg", filename="predicted.jpg")

# === Track Video Endpoint ===
@app.post("/track/video")
async def track_video(file: UploadFile = File(...)):
    logger.info("Received video for tracking")
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[-1]
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved input video to {input_path}")

    if model is None:
        return {"error": "Model not loaded"}

    results = model.track(source=input_path, save=True, save_txt=False, save_conf=True, project=OUTPUT_DIR, name=video_id)
    saved_path = os.path.join(OUTPUT_DIR, video_id, f"{video_id}.avi")
    logger.info(f"Tracking completed. Saved to {saved_path}")

    mime_type, _ = mimetypes.guess_type(saved_path)
    mime_type = mime_type or "application/octet-stream"

    return FileResponse(saved_path, media_type=mime_type, filename="tracked_video.avi")

# === Predict Video Endpoint ===
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    logger.info("Received video for prediction")
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[-1]
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved input video to {input_path}")

    if model is None:
        return {"error": "Model not loaded"}

    results = model.predict(source=input_path, save=True, project=OUTPUT_DIR, name=video_id, save_txt=False, save_conf=True)
    saved_path = os.path.join(OUTPUT_DIR, video_id, f"{video_id}.avi")
    logger.info(f"Prediction completed. Saved to {saved_path}")

    mime_type, _ = mimetypes.guess_type(saved_path)
    mime_type = mime_type or "application/octet-stream"

    return FileResponse(saved_path, media_type=mime_type, filename="predicted.avi")
