# Use a base image that is compatible with your Python version and offers apt-get
# Debian (bookworm) is what Azure App Service uses, so this is a good choice.
FROM python:3.13.3-slim-bookworm

# Set working directory
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy your requirements.txt and packages.txt first to leverage Docker caching
COPY requirements.txt .
COPY packages.txt .

# Install system dependencies from packages.txt (using apt-get directly)
# Ensure you include all necessary libGL, libXext, libSM libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install Python dependencies
# Use --no-cache-dir to reduce image size
# Crucially, we'll try to explicitly uninstall opencv-python if it gets pulled in by ultralytics
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy only required folders
COPY . .

# Expose the port your FastAPI app listens on
EXPOSE 8000

# Define the command to run your application
# This is your uvicorn command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
