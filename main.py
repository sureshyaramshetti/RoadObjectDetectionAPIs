import uuid
import os
import asyncio
import logging
import traceback
import mimetypes
import time
import shutil
import tempfile
from pathlib import Path
import cv2 # Import OpenCV for video frame processing and encoding
import io # For handling byte streams

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional, List, Union
from ultralytics import YOLO
from fastapi_utils.tasks import repeat_every 

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Global Data Structures ---
# Stores the status and result data for each job ID
# For a demo, in-memory dictionary is fine but will reset on app restart.
# In production, consider Redis or a database for persistent job status and a message queue for frames.
job_status: Dict[str, Dict[str, Union[str, None, List[bytes], int]]] = {} # Added List[bytes] for frames, int for total_frames

# --- Configuration for directories ---
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs" # Still used by YOLO for image predictions, but not for video output in streaming mode.

# Create base directories if they don't exist.
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Global Model Loading ---
model: Optional[YOLO] = None # Initialize as Optional
try:
    model_path = "model/best.pt"
    logger.info(f"Attempting to load model from {model_path}")
    model = YOLO(model_path)
    logger.info("Model loaded successfully.")
except Exception:
    logger.error("Failed to load model:")
    logger.error(traceback.format_exc())
    # Model remains None, health check will fail for processing endpoints

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup event triggered.")
    # Start the periodic cleanup task
    cleanup_old_files() 


# === Periodic Cleanup Task ===
@repeat_every(seconds=900) # every 15 minutes
def cleanup_old_files():
    """Deletes files and directories older than 30 minutes from UPLOAD_DIR and OUTPUT_DIR."""
    logger.info("Initiating periodic cleanup of old files and directories.")
    now = time.time()
    cutoff = now - 1800 # 30 minutes = 1800 seconds

    for base_dir in [UPLOAD_DIR, OUTPUT_DIR]:
        path_obj = Path(base_dir)
        if not path_obj.exists():
            logger.warning(f"Cleanup: Directory not found: {base_dir}")
            continue

        for item in path_obj.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    if item.stat().st_mtime < cutoff:
                        item.unlink()
                        logger.info(f"Deleted old file: {item}")
                elif item.is_dir():
                    # For directories, check if the directory itself is older than cutoff
                    # A more robust check might iterate contents, but this is simpler for temporary YOLO outputs
                    if item.stat().st_mtime < cutoff: 
                        shutil.rmtree(item)
                        logger.info(f"Deleted old directory: {item}")
            except Exception as e:
                logger.warning(f"Failed to delete {item} during cleanup: {e}")
    logger.info("Completed periodic cleanup.")


# --- Pydantic Models for API Responses ---
class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Video processing initiated."

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    # result_url now points to the streaming endpoint for videos, or a file download for images
    result_url: Optional[str] = None 
    error_message: Optional[str] = None
    total_frames: Optional[int] = None # For video streaming progress

class VideoUrlRequest(BaseModel):
    video_url: str
    mode: str = "track" # Can be "track" or "predict"

# --- Synchronous Helper for YOLO Processing ---
def _process_yolo_frame_sync(
    model_instance: YOLO, 
    source_path_or_url: str, # Can be a local path or a URL
    is_tracking: bool, 
    job_id: str,
    job_status_ref: Dict[str, Dict[str, Union[str, None, List[bytes], int]]]
):
    """
    Synchronous function to perform YOLO processing on video frames.
    This function will be run in a separate thread using asyncio.to_thread().
    """
    logger.info(f"[{job_id}] Starting synchronous YOLO processing thread for: {source_path_or_url}")
    
    cap_for_total_frames = None
    try:
        # Determine the YOLO method to call
        yolo_method = model_instance.track if is_tracking else model_instance.predict
        
        # Get total frames if possible (YOLO might not provide this directly from stream)
        # We use cv2.VideoCapture initially for total_frames count
        # This works for both local paths and URLs
        cap_for_total_frames = cv2.VideoCapture(source_path_or_url)
        if cap_for_total_frames.isOpened():
            total_frames = int(cap_for_total_frames.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_for_total_frames.release()
            job_status_ref[job_id]["total_frames"] = total_frames
            logger.info(f"[{job_id}] Video opened for total frame count. Total frames: {total_frames}")
        else:
            logger.warning(f"[{job_id}] Could not open video with cv2.VideoCapture to get total frame count from source: {source_path_or_url}. Total frames will be -1.")
            job_status_ref[job_id]["total_frames"] = -1 # Indicate unknown total frames

        frame_count = 0
        # Use YOLO's stream=True to process video frame by frame
        # This returns a generator that yields Results objects for each frame
        results_generator = yolo_method(source=source_path_or_url, stream=True, persist=True, verbose=False, imgsz=640)

        for results in results_generator: # Iterate over the results generator
            # Each 'results' object corresponds to a single frame
            
            # Draw results on the frame (YOLO's plot method returns a numpy array)
            annotated_frame = results.plot() # This draws boxes and labels

            # Encode the annotated frame to JPEG bytes for streaming
            ret_val, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret_val:
                logger.warning(f"[{job_id}] Failed to encode frame {frame_count} to JPEG.")
                continue
            
            # Append the JPEG bytes to the frames list
            # Ensure thread-safe access if multiple threads could write (though in this pattern, one thread per job)
            job_status_ref[job_id]["frames"].append(buffer.tobytes())
            job_status_ref[job_id]["processed_frames"] = frame_count + 1
            
            frame_count += 1
            if frame_count % 100 == 0: # Log progress every 100 frames
                logger.info(f"[{job_id}] Processed {frame_count} frames.")

        logger.info(f"[{job_id}] Video processing completed. Total frames processed: {frame_count}")
        job_status_ref[job_id]["status"] = "COMPLETED"

    except Exception as e:
        logger.error(f"[{job_id}] Error during video processing: {e}", exc_info=True)
        job_status_ref[job_id]["status"] = "FAILED"
        job_status_ref[job_id]["error_message"] = str(e)
    finally:
        if cap_for_total_frames and cap_for_total_frames.isOpened():
            cap_for_total_frames.release()


# --- Background Task Function for Video Processing (Async Wrapper) ---
async def process_video_in_background(job_id: str, source: str, is_tracking: bool):
    """
    This async function wraps the synchronous YOLO processing to run it in a separate thread.
    It manages the lifecycle of the background task and cleanup.
    The 'source' can be a local file path or a URL.
    """
    logger.info(f"[{job_id}] Initiating async background task for video processing from source: {source}")
    
    # Initialize job_status for streaming
    job_status[job_id].update({
        "status": "PROCESSING",
        "processed_frames": 0,
        "total_frames": 0,
        "frames": [], # List to store processed JPEG frames
        "error_message": None,
        "result_url": None # This will be set to the streaming URL
    })

    # Determine if source is a local file path for cleanup later
    is_local_file = Path(source).exists() # Check if it's a path that exists locally
    
    try:
        # Run the synchronous YOLO processing in a separate thread
        await asyncio.to_thread(
            _process_yolo_frame_sync, 
            model, 
            source, # Pass the source directly (path or URL)
            is_tracking, 
            job_id, 
            job_status # Pass reference to job_status for updates
        )

    except Exception as e:
        logger.error(f"[{job_id}] Unhandled error in background video processing: {e}", exc_info=True)
        job_status[job_id]["status"] = "FAILED"
        job_status[job_id]["error_message"] = str(e)
    finally:
        # Clean up temporary input file ONLY if it was a local upload
        if is_local_file and Path(source).exists():
            try:
                Path(source).unlink()
                logger.info(f"[{job_id}] Cleaned up local input video: {source}")
            except Exception as e:
                logger.warning(f"[{job_id}] Failed to delete local input video {source}: {e}")
        
        # Signal that all frames have been processed or an error occurred
        # No need to delete output_dir_for_yolo as we are not saving full video.


# --- Synchronous Helper for Image Prediction ---
def _predict_image_sync(
    model_instance: YOLO, 
    input_path: Path, 
    output_dir_for_request: Path
) -> Path:
    """
    Synchronous function to perform YOLO image prediction.
    This function will be run in a separate thread using asyncio.to_thread().
    Returns the path to the saved predicted image.
    """
    logger.info(f"Running synchronous image prediction thread for: {input_path}")
    
    # Ensure the output directory for this specific run exists before YOLO saves to it
    #os.makedirs(output_dir_for_request, exist_ok=True) 
    
    results = model_instance.predict(str(input_path), save=True, project=OUTPUT_DIR, name=str(output_dir_for_request.name), verbose=False)
    
    # --- Find the actual saved image file ---
    # YOLO saves to OUTPUT_DIR/image_id/original_filename.jpg
    # We need to construct the path based on this.
    saved_path = output_dir_for_request / Path(input_path).name
    
    if not saved_path.exists():
        # Fallback if YOLO renames or uses a different default
        found_files = list(output_dir_for_request.glob("*.jpg")) # or .png
        if found_files:
            saved_path = found_files[0]
        else:
            logger.error(f"Predicted image file not found at {saved_path} after processing. "
                         f"Contents of {output_dir_for_request}: {[p.name for p in output_dir_for_request.iterdir()] if output_dir_for_request.exists() else 'Directory not found'}")
            raise RuntimeError("Predicted image not found after processing. Check logs for details.")

    logger.info(f"Prediction complete. Output saved to {saved_path}")
    return saved_path


# --- API Endpoints ---
@app.get("/")
def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded.")
    return {"status": "Service Available", "message": "Road Object Detection Service is running and ready."}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Performs object detection on an uploaded image in a background thread.
    Returns the image with detected objects.
    """
    logger.info(f"Received request for image prediction: {file.filename}")
    image_id = str(uuid.uuid4())
    input_path = Path(UPLOAD_DIR) / f"{image_id}.jpg" 
    output_dir_for_request = Path(OUTPUT_DIR) / image_id 
    
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="YOLO model not loaded. Service unavailable for processing images.")

        # Save the uploaded image synchronously (file I/O is fast enough for images)
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Image saved to {input_path}")

        # Run prediction in a separate thread
        saved_path = await asyncio.to_thread(
            _predict_image_sync, 
            model, 
            input_path, 
            output_dir_for_request
        )

        return FileResponse(str(saved_path), media_type="image/jpeg", filename="predicted.jpg")
    except HTTPException: # Re-raise HTTPException if it was already raised
        raise
    except Exception as e:
        logger.error(f"Error during image prediction for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")
    finally:
        # --- Per-Request Cleanup for Input File ---
        if input_path.exists():
            try:
                input_path.unlink() # Delete the input file
                logger.info(f"Deleted input file: {input_path}")
            except Exception as e:
                logger.warning(f"Failed to delete input file {input_path}: {e}")
        # --- Note on Output File Cleanup ---
        # Output files (in output_dir_for_request) are returned via FileResponse.
        # They cannot be deleted immediately here because the client might still be downloading.
        # The periodic `cleanup_old_files` task will handle their deletion after 30 minutes.


@app.post("/process-video-async", response_model=JobResponse, status_code=202)
async def process_video_async(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(..., description="Video file to process with YOLO tracking/prediction."),
    mode: str = "track" # Can be "track" or "predict"
):
    """
    Initiates video processing (tracking or prediction) in the background and returns a job ID.
    The client should poll /status/{job_id} to get the result URL for streaming.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded. Service unavailable for video processing.")
    
    if mode not in ["track", "predict"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'track' or 'predict'.")

    job_id = str(uuid.uuid4())
    job_status[job_id] = {
        "status": "PENDING", 
        "result_url": f"/stream/{job_id}", # Point to the streaming endpoint immediately
        "error_message": None,
        "frames": [], # Initialize an empty list for frames
        "processed_frames": 0,
        "total_frames": 0
    }

    # Save the uploaded video to a temporary file
    ext = Path(file.filename).suffix
    input_path = Path(UPLOAD_DIR) / f"{job_id}{ext}"
    
    try:
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"[{job_id}] Received video '{file.filename}'. Saved to temporary path: {input_path}")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to save uploaded video: {e}", exc_info=True)
        job_status[job_id]["status"] = "FAILED"
        job_status[job_id]["error_message"] = f"Failed to save uploaded video: {e}"
        raise HTTPException(status_code=500, detail="Failed to save uploaded video.")

    is_tracking = (mode == "track")
    # Add the video processing task to the background
    background_tasks.add_task(process_video_in_background, job_id, str(input_path), is_tracking)

    return JobResponse(
        job_id=job_id,
        status="PENDING",
        message=f"Video {mode} initiated (from file upload). Use /status/{job_id} for progress and /stream/{job_id} for live feed."
    )

@app.post("/process-video-url-async", response_model=JobResponse, status_code=202)
async def process_video_url_async(
    request_body: VideoUrlRequest, # Use the Pydantic model for the request body
    background_tasks: BackgroundTasks
):
    """
    Initiates video processing (tracking or prediction) from a URL in the background and returns a job ID.
    The client should poll /status/{job_id} to get the result URL for streaming.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded. Service unavailable for video processing.")
    
    if request_body.mode not in ["track", "predict"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'track' or 'predict'.")

    job_id = str(uuid.uuid4())
    job_status[job_id] = {
        "status": "PENDING", 
        "result_url": f"/stream/{job_id}", # Point to the streaming endpoint immediately
        "error_message": None,
        "frames": [], # Initialize an empty list for frames
        "processed_frames": 0,
        "total_frames": 0
    }

    is_tracking = (request_body.mode == "track")
    # Pass the URL directly to the background task
    background_tasks.add_task(process_video_in_background, job_id, request_body.video_url, is_tracking)

    return JobResponse(
        job_id=job_id,
        status="PENDING",
        message=f"Video {request_body.mode} initiated (from URL). Use /status/{job_id} for progress and /stream/{job_id} for live feed."
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Checks the status of a video processing job.
    """
    status_info = job_status.get(job_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    return JobStatusResponse(
        job_id=job_id,
        status=status_info["status"],
        result_url=status_info["result_url"],
        error_message=status_info["error_message"],
        total_frames=status_info.get("total_frames"),
    )

@app.get("/stream/{job_id}")
async def stream_processed_video(job_id: str):
    """
    Streams processed video frames as Server-Sent Events (SSE).
    """
    status_info = job_status.get(job_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Job ID not found for streaming.")
    
    if status_info["status"] == "FAILED":
        raise HTTPException(status_code=500, detail=f"Video processing failed: {status_info.get('error_message', 'Unknown error')}")

    logger.info(f"[{job_id}] Client connected for streaming.")

    async def event_generator():
        sent_frames = 0
        while True:
            current_status = job_status[job_id]["status"]
            available_frames = len(job_status[job_id]["frames"])

            # Send new frames
            while sent_frames < available_frames:
                frame_data = job_status[job_id]["frames"][sent_frames]
                # SSE format: data: <payload>\n\n
                yield f"data: {frame_data.hex()}\n\n" # Send hex representation of bytes
                sent_frames += 1
                await asyncio.sleep(0.001) # Small pause to allow other tasks to run

            # Check if processing is complete or failed
            if current_status == "COMPLETED":
                logger.info(f"[{job_id}] Streaming completed. All frames sent.")
                break # Exit the loop, closing the SSE connection
            elif current_status == "FAILED":
                logger.error(f"[{job_id}] Streaming terminated due to processing failure.")
                yield f"event: error\ndata: {job_status[job_id].get('error_message', 'Processing failed')}\n\n"
                break
            
            # If not completed, wait a bit before checking for new frames
            await asyncio.sleep(0.1) # Poll for new frames every 100ms

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# The /results/{job_id}/{filename} endpoint is no longer needed for video streaming
# as frames are streamed directly. Keep it for image prediction if needed.
@app.get("/results/{job_id}/{filename}")
async def get_processed_file(job_id: str, filename: str):
    """
    Serves the processed image file from the local OUTPUT_DIR.
    This endpoint is primarily for image prediction results now.
    """
    # Construct the expected path to the processed image file
    file_path = Path(OUTPUT_DIR) / job_id / filename

    if not file_path.exists():
        logger.warning(f"Requested file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Processed file not found.")
    
    # Determine MIME type for the file
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "application/octet-stream" # Fallback

    logger.info(f"Serving processed file: {file_path} with MIME type {mime_type}")
    return FileResponse(str(file_path), media_type=mime_type, filename=filename)
