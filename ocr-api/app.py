from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Literal

# Model configuration
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
BASE_SIZE = int(os.getenv("DEEPSEEK_OCR_BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("DEEPSEEK_OCR_IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("DEEPSEEK_OCR_CROP_MODE", "true").lower() == "true"
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

app = FastAPI(title="DeepSeek-OCR API", version="2.0")
_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        t0 = time.time()
        print(f"[DeepSeek-OCR] Loading model {MODEL_NAME}...")
        
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        
        _model = _model.eval().cuda().to(torch.bfloat16)
        print(f"[DeepSeek-OCR] Ready in {time.time()-t0:.2f}s (GPU=True)")
    
    return _model, _tokenizer

# Size mode presets
SIZE_MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

class OCRResponse(BaseModel):
    text: str
    mode: str

class MarkdownResponse(BaseModel):
    markdown: str
    mode: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ocr/free", response_model=OCRResponse)
async def ocr_free(
    file: UploadFile = File(...),
    mode: Literal["tiny", "small", "base", "large", "gundam"] = Query("gundam", description="Size mode preset")
):
    """Simple OCR endpoint - extracts text from image"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image/* file")
    
    # Get model and tokenizer
    model, tokenizer = get_model()
    
    # Get size configuration
    config = SIZE_MODES[mode]
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Run inference
            prompt = "<image>\nFree OCR. "
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=tmp_file_path,
                output_path=None,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=False,
                test_compress=False
            )
            
            # Extract text from result (result format may vary, adjust as needed)
            text = result if isinstance(result, str) else str(result)
            
            return OCRResponse(text=text, mode=mode)
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

@app.post("/ocr/markdown", response_model=MarkdownResponse)
async def ocr_markdown(
    file: UploadFile = File(...),
    mode: Literal["tiny", "small", "base", "large", "gundam"] = Query("gundam", description="Size mode preset")
):
    """Markdown conversion endpoint - converts document to markdown format"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image/* file")
    
    # Get model and tokenizer
    model, tokenizer = get_model()
    
    # Get size configuration
    config = SIZE_MODES[mode]
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Run inference with markdown prompt
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=tmp_file_path,
                output_path=None,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=False,
                test_compress=False
            )
            
            # Extract markdown from result
            markdown = result if isinstance(result, str) else str(result)
            
            return MarkdownResponse(markdown=markdown, mode=mode)
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    file: UploadFile = File(...),
    mode: Literal["tiny", "small", "base", "large", "gundam"] = Query("gundam", description="Size mode preset"),
    prompt: Optional[str] = Query(None, description="Custom prompt (defaults to Free OCR)")
):
    """Configurable OCR endpoint with custom prompts"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image/* file")
    
    # Get model and tokenizer
    model, tokenizer = get_model()
    
    # Get size configuration
    config = SIZE_MODES[mode]
    
    # Use custom prompt or default
    ocr_prompt = prompt if prompt else "<image>\nFree OCR. "
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Run inference
            result = model.infer(
                tokenizer,
                prompt=ocr_prompt,
                image_file=tmp_file_path,
                output_path=None,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=False,
                test_compress=False
            )
            
            # Extract text from result
            text = result if isinstance(result, str) else str(result)
            
            return OCRResponse(text=text, mode=mode)
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
