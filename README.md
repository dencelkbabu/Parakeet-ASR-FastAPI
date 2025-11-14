
# Parakeet ASR

A high-performance Automatic Speech Recognition (ASR) server built with NVIDIA's NeMo Parakeet model. This application provides both REST API and WebSocket interfaces for transcribing audio files to text.

## 🚀 Key Features

-   **State-of-the-art ASR**: Powered by NVIDIA's parakeet-tdt-0.6b-v3 model (configurable via environment variables).
-   **⚡ GPU Acceleration & CUDA Graphs**: Fully optimized for CUDA, leveraging `cuda-python` for massive inference speed-ups via CUDA Graphs.
-   **Optimized Audio Loading**: Integrates fast audio decoding using the modern `torchcodec`, with intelligent fallback to standard `torchaudio`.
-   **Enhanced Monitoring**: New health checks report detailed runtime status, including device, model name, and the availability of key dependencies (`cuda-python`, `torchcodec`).
-   **Progress Logging**: Detailed, chunk-by-chunk processing status is logged to the console for long audio files, improving observability.
-   **Dynamic Resource Management**: Dynamically adjusts the ASR batch size based on available GPU memory during model loading.
-   **Multiple APIs**:
    -   REST API for simple file uploads (with chunked processing).
    -   WebSocket API for low-latency streaming transcription.
-   **Interactive Web UI**: Built-in browser interface for testing and demonstration.
-   **Docker Ready**: Easy deployment using Docker containers.

## 📋 Requirements
-   Python 3.10+
-   **NVIDIA GPU with CUDA 13.0+** (highly recommended for performance)
-   **FFmpeg DLLs** (required for Windows support, added via environment variable in the application)
-   Docker (technically optional, but recommended for deployment)    

## 🛠️ Installation

### Using Docker (Recommended) (Mac/Linux/Windows)

The easiest way to run Parakeet ASR is using Docker.

```
# Clone the repository
git clone [https://github.com/dencelkbabu/Parakeet-ASR-FastAPI.git](https://github.com/dencelkbabu/Parakeet-ASR-FastAPI.git)
cd parakeet-asr

# Build the Docker image
docker build -t parakeet-asr .

# Run the container (using --gpus all is crucial for GPU support)
docker run --gpus all -p 8777:8777 parakeet-asr

```

### Manual Installation (High-Performance CUDA)

This method requires manually installing specific PyTorch builds from the dedicated index to ensure CUDA support and compatibility with NeMo.

### Clone the repository
```
git clone https://github.com/dencelkbabu/Parakeet-ASR-FastAPI.git
cd parakeet-asr
```

### Create a virtual python environment
```
py -m venv .venv
.\.venv\Scripts\activate 
```
```
Use 'source .venv/bin/activate' on Linux/Mac
```

### --- STEP 1: Install CUDA-specific PyTorch packages ---
### **NOTE:** You MUST run this command separately before installing requirements.
```
pip install torch==2.9.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### --- STEP 2: Install remaining dependencies (NeMo, CUDA-Python, etc.) ---
### NeMo requires its own index URL to be installed correctly.
```
pip install -r app/requirements.txt --extra-index-url https://pypi.ngc.nvidia.com
```

### --- STEP 3: Run the application (Windows users may need to set FFmpeg path) ---
```
cd app
python main.py
```
On Windows, you might need to set the environment variable for **FFmpeg** DLL loading  
Preferred: [ffmpeg-7.1.1-full_build-shared.7z](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-full_build-shared.7z)

## ⚙️ Configuration
The application can be configured using environment variables or a `.env` file in the app directory:

| Variable | Description | Default |
|----------|-------------|---------|
| ASR_MODEL_NAME | Name of the **NeMo** model to load. | `nvidia/parakeet-tdt-0.6b-v3`
| `BATCH_SIZE` | ASR batch size during inference. Adjusted dynamically based on GPU memory if too high. | 1 |
| `NUM_WORKERS` | Number of workers for processing | 0 |
| `TRANSCRIBE_CHUNK_LEN` | Audio chunk length in seconds | 30 |
| `TRANSCRIBE_OVERLAP` | Overlap between chunks in seconds | 5 |
| `SAMPLE_RATE` | Audio sample rate (Hz) | 16000 |
| `PORT` | Server port | 8777 |
| `LOG_LEVEL` | Logging level | INFO |

## 🔌 API Documentation

### REST API

#### Transcribe Audio

```
POST /v1/audio/transcriptions
```

**Key Enhancement:** The server now provides verbose progress logging for long files in the console.

**Request:**

-   Content-Type: `multipart/form-data`
-   Body: Form with an audio file attached as `file`

**Response:**
```
{
  "text": "The complete transcribed text.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.55,
      "text": "This is the first segment."
    },
    {
      "id": 1,
      "start": 2.56,
      "end": 5.2,
      "text": "This is the second segment."
    }
  ],
  "language": "en",
  "transcription_time": 1.234
}
```

#### Health Check (`GET /health`)
**New:** This endpoint now includes dependency status and model information.

**Response:**
```
{
  "status": "ok",
  "device": "cuda",
  "gpu_available": true,
  "model_loaded": true,
  "current_asr_model": "nvidia/parakeet-tdt-0.6b-v3",
  "cuda_python_installed": true,
  "torchcodec_available": true
}
```

### WebSocket API
Connect to `/v1/audio/transcriptions/ws` endpoint.

**Key Enhancement:** The server logs **chunk-by-chunk processing status** to the console, and the client receives segment results almost instantly as they are processed.

1.  **Connection**: Connect to the WebSocket endpoint.
2.  **Configuration**: Send a JSON message with audio configuration:
    ```
    {
      "sample_rate": 16000,
      "channels": 1,
      "format": "binary"
    }
    ```  
3.  **Audio Data**: Send audio data in binary chunks.
4.  **End Signal**: Send `"END"` as a text message to signal the end of the audio stream.
5.  **Receiving Results**: The server sends JSON messages for each transcribed segment as they become available.
6.  **Final Result**: After processing, a summary message with the full transcription is sent.
    

## 🖥️ Web Interface

A web interface is available at the root URL (`/`). This provides a simple way to test the transcription services:

-   Upload audio files
-   Choose between REST and WebSocket APIs
-   View transcription results and timing information
-   Debug mode for detailed logging (now includes chunk progress)
    

## 🛠️ Development

To set up a development environment:

### Clone the repository
```
git clone https://github.com/dencelkbabu/Parakeet-ASR-FastAPI.git
cd parakeet-asr
```
### Create a separate development environment
```
py -m venv .venv-dev
.\.venv-dev\Scripts\activate 
```
```
# Use 'source .venv-dev/bin/activate' on Linux/Mac
```
### Follow the manual installation steps (Steps 1 & 2) above for CUDA setup.
### Run with debug logging (highly recommended to see chunk progress)
```
cd app
LOG_LEVEL=DEBUG python main.py
```

## 🔍 Troubleshooting

Common issues:
-   **Windows FFmpeg Error**: If audio loading fails on Windows, ensure your FFmpeg installation path (e.g., `C:\ffmpeg\bin`) is in your system's `PATH` environment variable. The application now attempts a fix, but this is the ultimate manual solution.
-   **Model Loading Errors**: Ensure you have enough GPU memory available. The application now attempts to adjust the `BATCH_SIZE` dynamically.
-   **Performance Issues**: If transcription is slow, verify the `/health` endpoint reports `cuda_python_installed: true` and a CUDA device is in use.
-   **Audio Format Issues**: The application works best with WAV files but supports various formats through torchaudio.
    

## 📄 License
This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE "null").

## 🙏 Acknowledgements
-   This project is a fork of and based on the original: [Parakeet-ASR-FastAPI](https://github.com/pnivek/Parakeet-ASR-FastAPI "null") by [Kevin](https://github.com/pnivek "null").
-   Special thanks to the original author for their work.
-   [NVIDIA NeMo](https://github.com/NVIDIA/NeMo "null") for the Parakeet ASR model
-   [FastAPI](https://fastapi.tiangolo.com/ "null") for the web framework
-   [PyTorch](https://pytorch.org/ "null") and [torchaudio](https://pytorch.org/audio "null") for audio processing