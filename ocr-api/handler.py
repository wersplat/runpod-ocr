import runpod
import base64
import tempfile
import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch
import json

# Model configuration
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
BASE_SIZE = int(os.getenv("DEEPSEEK_OCR_BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("DEEPSEEK_OCR_IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("DEEPSEEK_OCR_CROP_MODE", "true").lower() == "true"

# Global model and tokenizer
_model = None
_tokenizer = None

# Size mode presets
SIZE_MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

def get_model():
    """Load and cache the model"""
    global _model, _tokenizer
    if _model is None:
        import time
        import sys
        t0 = time.time()
        print(f"[DeepSeek-OCR] Starting model load: {MODEL_NAME}", flush=True)
        sys.stdout.flush()
        
        try:
            print("[DeepSeek-OCR] Loading tokenizer...", flush=True)
            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True
            )
            print("[DeepSeek-OCR] Tokenizer loaded successfully", flush=True)
            
            # Try flash_attention_2 first, fallback to standard attention
            try:
                print("[DeepSeek-OCR] Attempting to load model with flash_attention_2...", flush=True)
                _model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True
                )
                print("[DeepSeek-OCR] Using flash_attention_2", flush=True)
            except Exception as e:
                print(f"[DeepSeek-OCR] flash_attention_2 not available ({e}), using standard attention", flush=True)
                _model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            
            print("[DeepSeek-OCR] Moving model to GPU...", flush=True)
            _model = _model.eval().cuda().to(torch.bfloat16)
            print(f"[DeepSeek-OCR] Model ready in {time.time()-t0:.2f}s (GPU=True)", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"[DeepSeek-OCR] ERROR loading model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
    
    return _model, _tokenizer

def process_image(image_data, mode="gundam", prompt=None):
    """Process image with DeepSeek-OCR"""
    model, tokenizer = get_model()
    
    # Get size configuration
    config = SIZE_MODES.get(mode, SIZE_MODES["gundam"])
    
    # Determine prompt
    if prompt:
        ocr_prompt = prompt
    else:
        # Default prompt based on mode or use Free OCR
        ocr_prompt = "<image>\nFree OCR. "
    
    # Handle different input formats
    if isinstance(image_data, str):
        # Check if it's a URL
        if image_data.startswith(("http://", "https://")):
            import requests
            response = requests.get(image_data)
            response.raise_for_status()
            image_bytes = response.content
        # Check if it's a data URL (data:image/...)
        elif image_data.startswith("data:image"):
            # Extract base64 part after comma
            image_bytes = base64.b64decode(image_data.split(",", 1)[1])
        else:
            # Assume it's base64 encoded
            image_bytes = base64.b64decode(image_data)
    else:
        # Assume it's already bytes
        image_bytes = image_data
    
    # Save image to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        try:
            tmp_file.write(image_bytes)
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
            
            return text
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

def handler(event):
    """
    RunPod serverless handler
    
    Expected input format:
    {
        "input": {
            "image": "<base64_encoded_image> or <image_url>",
            "mode": "gundam",  # optional: tiny, small, base, large, gundam
            "prompt": "<custom_prompt>",  # optional
            "format": "text" or "markdown"  # optional: determines default prompt
        }
    }
    """
    import sys
    import json
    
    print(f"[DeepSeek-OCR] ========== HANDLER CALLED ==========", flush=True)
    print(f"[DeepSeek-OCR] Handler called with event type: {type(event)}", flush=True)
    print(f"[DeepSeek-OCR] Event keys: {list(event.keys()) if isinstance(event, dict) else 'not a dict'}", flush=True)
    print(f"[DeepSeek-OCR] Full event (first 500 chars): {str(event)[:500]}", flush=True)
    sys.stdout.flush()
    
    try:
        # Handle different event formats RunPod might send
        if isinstance(event, dict):
            input_data = event.get("input", event)  # Try "input" key, fallback to event itself
        else:
            print(f"[DeepSeek-OCR] Event is not a dict, type: {type(event)}", flush=True)
            input_data = {}
        
        print(f"[DeepSeek-OCR] Input data type: {type(input_data)}", flush=True)
        print(f"[DeepSeek-OCR] Input data keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'not a dict'}", flush=True)
        sys.stdout.flush()
        
        # Get image data
        image_data = input_data.get("image")
        if not image_data:
            return {
                "error": "Missing 'image' field in input. Provide base64 encoded image, data URL, or image URL."
            }
        
        # Get mode (default: gundam)
        mode = input_data.get("mode", "gundam")
        if mode not in SIZE_MODES:
            mode = "gundam"
        
        # Get format (text or markdown)
        format_type = input_data.get("format", "text")
        
        # Determine prompt based on format
        prompt = input_data.get("prompt")
        if not prompt:
            if format_type == "markdown":
                prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            else:
                prompt = "<image>\nFree OCR. "
        
        # Process image
        result_text = process_image(image_data, mode=mode, prompt=prompt)
        
        # Return result
        if format_type == "markdown":
            return {
                "markdown": result_text,
                "mode": mode
            }
        else:
            return {
                "text": result_text,
                "mode": mode
            }
    
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__
        }

# Start RunPod serverless
if __name__ == "__main__":
    import sys
    import os
    
    print("[DeepSeek-OCR] ========================================", flush=True)
    print("[DeepSeek-OCR] Starting RunPod serverless handler...", flush=True)
    print(f"[DeepSeek-OCR] Python version: {sys.version}", flush=True)
    print(f"[DeepSeek-OCR] Working directory: {os.getcwd()}", flush=True)
    print(f"[DeepSeek-OCR] Files in /app: {os.listdir('/app')}", flush=True)
    print(f"[DeepSeek-OCR] Handler function: {handler}", flush=True)
    print(f"[DeepSeek-OCR] Handler type: {type(handler)}", flush=True)
    print(f"[DeepSeek-OCR] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DeepSeek-OCR] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[DeepSeek-OCR] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)
    
    # Test handler registration by calling it with a test event
    print("[DeepSeek-OCR] Testing handler registration...", flush=True)
    try:
        test_event = {"input": {"image": "test", "mode": "gundam", "format": "text"}}
        # This will fail because image is invalid, but we'll see if handler is callable
        test_result = handler(test_event)
        print(f"[DeepSeek-OCR] Handler test call returned: {type(test_result)}", flush=True)
    except Exception as test_err:
        print(f"[DeepSeek-OCR] Handler test call failed (expected): {test_err}", flush=True)
    
    print("[DeepSeek-OCR] ========================================", flush=True)
    print("[DeepSeek-OCR] Registering handler with RunPod...", flush=True)
    sys.stdout.flush()
    
    try:
        # Check RunPod SDK version
        print(f"[DeepSeek-OCR] RunPod SDK version: {runpod.__version__ if hasattr(runpod, '__version__') else 'unknown'}", flush=True)
        print(f"[DeepSeek-OCR] RunPod module path: {runpod.__file__}", flush=True)
        print(f"[DeepSeek-OCR] RunPod serverless module: {runpod.serverless}", flush=True)
        sys.stdout.flush()
        
        print("[DeepSeek-OCR] Calling runpod.serverless.start()...", flush=True)
        sys.stdout.flush()
        
        # Register handler - RunPod expects this exact format
        # The handler function should be passed directly, not as a string
        runpod.serverless.start({"handler": handler})
        
        print("[DeepSeek-OCR] runpod.serverless.start() returned (unexpected - should block forever)", flush=True)
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("[DeepSeek-OCR] Interrupted by user", flush=True)
        sys.stdout.flush()
        raise
    except Exception as e:
        print(f"[DeepSeek-OCR] FATAL ERROR starting serverless: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

