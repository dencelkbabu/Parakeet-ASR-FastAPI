FROM python:3.10-slim

WORKDIR /app

# Install system dependencies: FFmpeg and libsndfile1 are critical for torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    # Clears package lists to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY app/requirements.txt /app/

# --- INSTALL PYTHON DEPENDENCIES ---
# This step is updated to correctly install the CUDA-optimized stack.
# We must use two external indices:
# 1. https://download.pytorch.org/whl/cu130 for the PyTorch/Torchaudio CUDA binaries.
# 2. https://pypi.ngc.nvidia.com for NeMo-toolkit components.
RUN pip install --no-cache-dir \
    # 1. Install PyTorch/Torchaudio explicitly with their CUDA index
    torch==2.9.1+cu130 torchaudio==2.9.1+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    \
    # 2. Install the remaining dependencies from requirements.txt 
    # (including nemo-toolkit, cuda-python, torchcodec, etc.)
    && pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://pypi.ngc.nvidia.com

# Copy application files (main.py and other source files in 'app/')
COPY app/ /app/

# Create static directory (or ensure it exists)
RUN mkdir -p /app/static

# Assuming 'index.html' is at the root of the build context, copy it 
# into the expected static directory for the FastAPI app.
COPY index.html /app/static/

# Expose the port defined in the .env file (default: 8777)
EXPOSE 8777

# Ensure output is unbuffered, which helps in Docker logging
ENV PYTHONUNBUFFERED=1

# Start the application
# We rely on main.py to run uvicorn with host="0.0.0.0"
CMD ["python", "main.py"]