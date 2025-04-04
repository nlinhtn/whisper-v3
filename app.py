import os
import tempfile
import torch
import whisper
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper Model
print("Loading Whisper large-v3 model...")
model = whisper.load_model("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
print(f"Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")

@app.get("/")
def read_root():
    return {"status": "Whisper API is running", "gpu_available": torch.cuda.is_available()}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    try:
        # Thực hiện nhận dạng
        result = model.transcribe(temp_file_path, language="vi")

        # Xóa file tạm sau khi xử lý
        os.unlink(temp_file_path)

        return {"text": result["text"], "segments": result["segments"]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Streaming API endpoint - cho realtime recognition
@app.post("/transcribe-chunk/")
async def transcribe_chunk(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    try:
        # Chuyển đổi options cho realtime (low latency)
        options = {
            "beam_size": 5,
            "best_of": 5,
            "fp16": torch.cuda.is_available()
        }
        result = model.transcribe(temp_file_path, language="vi", **options)

        return {"text": result["text"]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)