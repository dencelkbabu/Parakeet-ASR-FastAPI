import os
import io
import time
import shutil
import tempfile
import base64
import asyncio
import logging
# --- BEGIN MONKEY-PATCH FOR WINDOWS ---
import signal
# NeMo's exp_manager (used by nemo.collections.asr) relies on signal.SIGKILL,
# which does not exist on Windows. We patch the 'signal' module to map
# SIGKILL to SIGTERM as a workaround to allow the import to succeed.
if os.name == 'nt' and not hasattr(signal, 'SIGKILL'):
    setattr(signal, 'SIGKILL', signal.SIGTERM)
# --- END MONKEY-PATCH FOR WINDOWS ---
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Set device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("parakeet-asr")
logger.info(f"Using device: {device}")

# model configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
CHUNK_LENGTH = float(os.getenv("TRANSCRIBE_CHUNK_LEN", 30))
OVERLAP = float(os.getenv("TRANSCRIBE_OVERLAP", 5))
MODEL_SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
PORT = int(os.getenv("PORT", 8777))
logger.info(f"Model config: BATCH_SIZE: {BATCH_SIZE}, NUM_WORKERS: {NUM_WORKERS}, CHUNK_LENGTH: {CHUNK_LENGTH}, OVERLAP: {OVERLAP}, MODEL_SAMPLE_RATE: {MODEL_SAMPLE_RATE}, DEVICE: {device}")

# FAST API app with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
        logger.warning(f"Created static directory at {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Root endpoint to serve index.html
@app.get("/", response_class=HTMLResponse)
async def get_index():
    app_dir = os.path.dirname(__file__)
    index_path = os.path.join(app_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.error(f"Index file not found at {index_path}")
        return HTMLResponse(content="<html><body><h1>Parakeet ASR Server</h1><p>UI not available. index.html not found.</p></body></html>")

# load model
asr_model = None
try:
    logger.info("Loading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
    asr_model = asr_model.to(device)
    asr_model.preprocessor.featurizer.dither = 0.0
    logger.info("ASR model loaded successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not load ASR model. Error: {e}")
    asr_model = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": asr_model is not None
    }

def run_asr_on_tensor_chunk(
    audio_chunk_tensor: torch.Tensor,
    chunk_time_offset: float
) -> tuple[list[dict], list[str]]:
    if not asr_model:
        logger.error("run_asr_on_tensor_chunk: ASR model is not loaded.")
        return [], []
    assert audio_chunk_tensor.ndim == 1, f"ERROR: run_asr_on_tensor_chunk received tensor with ndim={audio_chunk_tensor.ndim}. Expected 1D."
    assert audio_chunk_tensor.numel() > 0, "ERROR: run_asr_on_tensor_chunk received an empty audio tensor."
    try:
        audio_chunk_tensor = audio_chunk_tensor.to(device)
        output_hypotheses: list = asr_model.transcribe(
            audio=[audio_chunk_tensor],
            batch_size=BATCH_SIZE,
            return_hypotheses=True,
            timestamps=True,
            verbose=False,
            num_workers=NUM_WORKERS
        )
        processed_segments = []
        chunk_text_list = []
        if not (output_hypotheses and isinstance(output_hypotheses, list) and len(output_hypotheses) == 1):
            logger.warning(f"run_asr_on_tensor_chunk: Unexpected output format from ASR model: {type(output_hypotheses)}")
            return [], []
        hypothesis = output_hypotheses[0]
        if hasattr(hypothesis, 'text') and hypothesis.text:
            transcribed_text = hypothesis.text.strip()
            if transcribed_text:
                chunk_text_list.append(transcribed_text)
        if hasattr(hypothesis, 'timestamp') and hypothesis.timestamp:
            segment_metadata_list = hypothesis.timestamp.get("segment", [])
            for seg_meta in segment_metadata_list:
                seg_text = seg_meta.get("segment", "").strip()
                if not seg_text:
                    continue
                abs_start_time = round(seg_meta["start"] + chunk_time_offset, 3)
                abs_end_time = round(seg_meta["end"] + chunk_time_offset, 3)
                processed_segments.append({
                    "start": abs_start_time,
                    "end": abs_end_time,
                    "text": seg_text,
                    "id": 0,
                    "seek": 0,
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": None,
                    "compression_ratio": None,
                    "no_speech_prob": None
                })
        return processed_segments, chunk_text_list
    except Exception as e:
        logger.error(f"run_asr_on_tensor_chunk: Exception during ASR processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []

@app.post("/v1/audio/transcriptions")
async def transcribe_rest(file: UploadFile = File(...)):
    if not asr_model:
        logger.error("transcribe_rest: ASR model not available.")
        return JSONResponse(status_code=503, content={"error": "ASR model not available."})
    uploaded_full_temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_full_audio:
            shutil.copyfileobj(file.file, tmp_full_audio)
            uploaded_full_temp_file_path = tmp_full_audio.name
        try:
            waveform_full, sr_orig = torchaudio.load(uploaded_full_temp_file_path)
        except Exception as load_err:
            logger.error(f"transcribe_rest: Failed to load audio from {uploaded_full_temp_file_path}: {load_err}")
            raise
        if sr_orig != MODEL_SAMPLE_RATE:
            waveform_full = torchaudio.functional.resample(waveform_full, orig_freq=sr_orig, new_freq=MODEL_SAMPLE_RATE)
        if waveform_full.ndim > 1 and waveform_full.shape[0] > 1:
            waveform_full = waveform_full.mean(dim=0, keepdim=True)
        elif waveform_full.ndim == 1:
             waveform_full = waveform_full.unsqueeze(0)
        if waveform_full.shape[1] == 0:
            logger.info("transcribe_rest: Audio content is empty after loading and preprocessing.")
            return JSONResponse(content={"text": "", "segments": [], "language": "en", "transcription_time": 0.0})
        waveform_full = waveform_full.to(device)
        total_duration_seconds = waveform_full.shape[1] / MODEL_SAMPLE_RATE
        start_time_processing = time.time()
        current_processing_time_seconds = 0.0
        all_segments = []
        while current_processing_time_seconds < total_duration_seconds:
            actual_chunk_start_seconds = max(0, current_processing_time_seconds - OVERLAP)
            actual_chunk_end_seconds = min(total_duration_seconds, current_processing_time_seconds + CHUNK_LENGTH)
            start_sample = int(actual_chunk_start_seconds * MODEL_SAMPLE_RATE)
            end_sample = int(actual_chunk_end_seconds * MODEL_SAMPLE_RATE)
            if start_sample >= end_sample:
                break
            chunk_slice_2d = waveform_full[:, start_sample:end_sample]
            audio_chunk_for_asr = chunk_slice_2d.squeeze(0)
            if audio_chunk_for_asr.numel() == 0:
                current_processing_time_seconds += (CHUNK_LENGTH - OVERLAP)
                continue
            segments_from_chunk, _ = await asyncio.to_thread(
                run_asr_on_tensor_chunk,
                audio_chunk_for_asr,
                actual_chunk_start_seconds
            )
            for seg_meta in segments_from_chunk:
                if seg_meta["start"] >= current_processing_time_seconds or not all_segments:
                    all_segments.append({
                        "id": len(all_segments),
                        "start": seg_meta["start"],
                        "end": seg_meta["end"],
                        "text": seg_meta["text"],
                        "seek": 0,
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": None,
                        "compression_ratio": None,
                        "no_speech_prob": None
                    })
            current_processing_time_seconds += (CHUNK_LENGTH - OVERLAP)
        transcription_duration_seconds = round(time.time() - start_time_processing, 3)
        final_text = " ".join(s['text'] for s in all_segments).strip()
        result = {
            "text": final_text, "segments": all_segments, "language": "en",
            "transcription_time": transcription_duration_seconds
        }
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"transcribe_rest: Unhandled exception: {str(e)}")
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Failed to process audio.", "detail": str(e)})
    finally:
        if uploaded_full_temp_file_path and os.path.exists(uploaded_full_temp_file_path):
            os.remove(uploaded_full_temp_file_path)

@app.websocket("/v1/audio/transcriptions/ws")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    if not asr_model:
        logger.error("websocket_transcribe: ASR model not available.")
        await websocket.send_json({"error": "ASR model not available."})
        await websocket.close(code=1011)
        return
    main_audio_buffer = bytearray()
    client_config = {}
    is_connected = True
    try:
        try:
            config_message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            client_config.update(config_message)
            logger.info(f"WebSocket (/ws) client reported config: {client_config}")
        except asyncio.TimeoutError:
            logger.info("WebSocket (/ws): Client configuration timeout. Proceeding.")
        except Exception as e:
            logger.info(f"WebSocket (/ws): Error receiving client configuration: {e}. Proceeding.")
        logger.info("WebSocket (/ws): Waiting for audio data stream from client...")
        while is_connected:
            try:
                if websocket.application_state != WebSocketState.CONNECTED:
                    is_connected = False
                    break
                message = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                if "bytes" in message:
                    main_audio_buffer.extend(message["bytes"])
                elif "text" in message:
                    if client_config.get("format") == "base64":
                        try:
                            main_audio_buffer.extend(base64.b64decode(message["text"]))
                        except Exception as b64e:
                            logger.warning(f"WebSocket (/ws): Base64 decode error: {b64e}")
                    elif message["text"].upper() == "END":
                        logger.info(f"WebSocket (/ws): END signal received. Total bytes: {len(main_audio_buffer)}")
                        is_connected = False
                        break
            except asyncio.TimeoutError:
                logger.warning("WebSocket (/ws): Timeout waiting for message. Assuming stream ended or client stalled.")
                is_connected = False
                break
            except WebSocketDisconnect:
                logger.info("WebSocket (/ws): Client disconnected during data accumulation.")
                is_connected = False
                break
            except Exception as e:
                logger.error(f"WebSocket (/ws): Exception in receive loop: {str(e)}")
                import traceback; traceback.print_exc()
                is_connected = False
                break
        if not main_audio_buffer:
            logger.info("WebSocket (/ws): No audio data received.")
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({"error": "No audio data received", "type": "error"})
            return
        logger.info("WebSocket (/ws): Starting full audio processing.")
        processing_start_time = time.time()
        all_segments_sent_to_client = []
        try:
            audio_io_buffer = io.BytesIO(main_audio_buffer)
            full_waveform, sr_original = torchaudio.load(audio_io_buffer)
            main_audio_buffer.clear()
            logger.info(f"WebSocket (/ws): Audio decoded. Original SR={sr_original}, Shape={full_waveform.shape}")
            if sr_original != MODEL_SAMPLE_RATE:
                full_waveform = torchaudio.functional.resample(full_waveform, orig_freq=sr_original, new_freq=MODEL_SAMPLE_RATE)
            if full_waveform.ndim > 1 and full_waveform.shape[0] > 1:
                full_waveform = full_waveform.mean(dim=0)
            elif full_waveform.ndim == 2 and full_waveform.shape[0] == 1:
                full_waveform = full_waveform.squeeze(0)
            full_waveform = full_waveform.to(device)
            if full_waveform.numel() == 0:
                logger.info("WebSocket (/ws): Audio content is empty after decoding/preprocessing.")
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "text": "", "segments": [], "language": "en",
                        "transcription_time": 0.0, "total_segments": 0,
                        "final_duration_processed_seconds": 0.0, "type": "final_transcription"
                    })
                return
            total_audio_duration_seconds = full_waveform.shape[0] / MODEL_SAMPLE_RATE
            current_processing_window_start_seconds = 0.0
            logger.info(f"WebSocket (/ws): Server-side chunking. Total Duration: {total_audio_duration_seconds:.2f}s. ChunkLen: {CHUNK_LENGTH}s, Overlap: {OVERLAP}s")
            while current_processing_window_start_seconds < total_audio_duration_seconds:
                if websocket.application_state != WebSocketState.CONNECTED:
                    logger.info("WebSocket (/ws): Client disconnected during chunk processing.")
                    break
                actual_asr_chunk_start_seconds = max(0, current_processing_window_start_seconds - OVERLAP)
                actual_asr_chunk_end_seconds = min(total_audio_duration_seconds, current_processing_window_start_seconds + CHUNK_LENGTH)
                start_sample_idx = int(actual_asr_chunk_start_seconds * MODEL_SAMPLE_RATE)
                end_sample_idx = int(actual_asr_chunk_end_seconds * MODEL_SAMPLE_RATE)
                if start_sample_idx >= end_sample_idx:
                    break
                audio_chunk_for_asr = full_waveform[start_sample_idx:end_sample_idx]
                if audio_chunk_for_asr.numel() == 0:
                    current_processing_window_start_seconds += (CHUNK_LENGTH - OVERLAP)
                    continue
                segments_from_chunk, _ = await asyncio.to_thread(
                    run_asr_on_tensor_chunk,
                    audio_chunk_for_asr,
                    actual_asr_chunk_start_seconds
                )
                if websocket.application_state == WebSocketState.CONNECTED:
                    for segment_data in segments_from_chunk:
                        if segment_data["start"] >= current_processing_window_start_seconds or not all_segments_sent_to_client:
                            segment_data["id"] = len(all_segments_sent_to_client)
                            await websocket.send_json(segment_data)
                            all_segments_sent_to_client.append(segment_data)
                else:
                    logger.info("WebSocket (/ws): Client disconnected while sending segments.")
                    break
                current_processing_window_start_seconds += (CHUNK_LENGTH - OVERLAP)
            if websocket.application_state == WebSocketState.CONNECTED:
                final_transcription_text = " ".join(s['text'] for s in all_segments_sent_to_client).strip()
                transcription_duration = round(time.time() - processing_start_time, 3)
                await websocket.send_json({
                    "text": final_transcription_text,
                    "language": "en",
                    "transcription_time": transcription_duration,
                    "total_segments": len(all_segments_sent_to_client),
                    "final_duration_processed_seconds": round(total_audio_duration_seconds, 3),
                    "type": "final_transcription"
                })
            logger.info("WebSocket (/ws): All processing complete.")
        except Exception as audio_processing_error:
            logger.error(f"WebSocket (/ws): Error during audio processing phase: {audio_processing_error}")
            import traceback; traceback.print_exc()
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({"error": f"Audio processing error: {audio_processing_error}", "type": "error"})
    except Exception as outer_exception:
        logger.error(f"WebSocket (/ws): Unhandled exception in handler: {str(outer_exception)}")
        import traceback; traceback.print_exc()
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
        logger.info("WebSocket (/ws) handler finished.")

if __name__ == "__main__":
    import uvicorn
    if not asr_model:
        logger.critical("Cannot start server: ASR Model failed to load.")
    else:
        logger.info(f"Starting server on port {PORT}. Model Rate: {MODEL_SAMPLE_RATE}")
        uvicorn.run(app, host="0.0.0.0", port=PORT)
