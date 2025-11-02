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
    import sys
    
    print(f"[DeepSeek-OCR] process_image() called with mode={mode}, prompt={'set' if prompt else 'None'}", flush=True)
    sys.stdout.flush()
    
    try:
        print(f"[DeepSeek-OCR] Getting model...", flush=True)
        sys.stdout.flush()
        model, tokenizer = get_model()
        print(f"[DeepSeek-OCR] Model retrieved successfully", flush=True)
        sys.stdout.flush()
        
        # Get size configuration
        config = SIZE_MODES.get(mode, SIZE_MODES["gundam"])
        print(f"[DeepSeek-OCR] Using config: base_size={config['base_size']}, image_size={config['image_size']}, crop_mode={config['crop_mode']}", flush=True)
        
        # Determine prompt
        if prompt:
            ocr_prompt = prompt
        else:
            # Default prompt based on mode or use Free OCR
            ocr_prompt = "<image>\nFree OCR. "
        print(f"[DeepSeek-OCR] Using OCR prompt: {ocr_prompt[:50]}...", flush=True)
        sys.stdout.flush()
        
        # Validate image_data
        if image_data is None:
            raise ValueError("image_data cannot be None")
        
        print(f"[DeepSeek-OCR] Processing image_data, type: {type(image_data)}", flush=True)
        sys.stdout.flush()
        
        # Handle different input formats
        image_bytes = None
        if isinstance(image_data, str):
            print(f"[DeepSeek-OCR] Image data is string, length: {len(image_data)}", flush=True)
            # Check if it's a URL
            if image_data.startswith(("http://", "https://")):
                print(f"[DeepSeek-OCR] Detected URL, fetching...", flush=True)
                import requests
                response = requests.get(image_data)
                response.raise_for_status()
                image_bytes = response.content
                print(f"[DeepSeek-OCR] Fetched {len(image_bytes)} bytes from URL", flush=True)
            # Check if it's a data URL (data:image/...)
            elif image_data.startswith("data:image"):
                print(f"[DeepSeek-OCR] Detected data URL, decoding...", flush=True)
                # Extract base64 part after comma
                try:
                    image_bytes = base64.b64decode(image_data.split(",", 1)[1])
                    print(f"[DeepSeek-OCR] Decoded {len(image_bytes)} bytes from data URL", flush=True)
                except (IndexError, ValueError) as e:
                    print(f"[DeepSeek-OCR] ERROR decoding data URL: {e}", flush=True)
                    raise ValueError(f"Invalid data URL format: {e}")
            else:
                # Assume it's base64 encoded
                print(f"[DeepSeek-OCR] Assuming base64 string, decoding...", flush=True)
                try:
                    image_bytes = base64.b64decode(image_data)
                    print(f"[DeepSeek-OCR] Decoded {len(image_bytes)} bytes from base64", flush=True)
                except Exception as e:
                    print(f"[DeepSeek-OCR] ERROR decoding base64: {e}", flush=True)
                    raise ValueError(f"Failed to decode base64 image data: {e}")
        elif isinstance(image_data, bytes):
            print(f"[DeepSeek-OCR] Image data is bytes, length: {len(image_data)}", flush=True)
            image_bytes = image_data
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}. Expected str (base64/URL) or bytes.")
        
        sys.stdout.flush()
        
        # Validate image_bytes
        if not image_bytes:
            raise ValueError("image_bytes is empty after processing")
        
        if len(image_bytes) == 0:
            raise ValueError("image_bytes has zero length")
        
        print(f"[DeepSeek-OCR] Image bytes validated: {len(image_bytes)} bytes", flush=True)
        sys.stdout.flush()
        
        # Save image to temporary file
        tmp_file_path = None
        output_dir = None
        try:
            print(f"[DeepSeek-OCR] Creating temporary file...", flush=True)
            sys.stdout.flush()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_file_path = tmp_file.name
                print(f"[DeepSeek-OCR] Temporary file created: {tmp_file_path}", flush=True)
            
            # Validate temp file path
            if not tmp_file_path:
                raise ValueError("Failed to create temporary file: tmp_file_path is None")
            
            if not os.path.exists(tmp_file_path):
                raise ValueError(f"Temporary file was not created: {tmp_file_path}")
            
            # Verify file has content
            file_size = os.path.getsize(tmp_file_path)
            if file_size == 0:
                raise ValueError(f"Temporary file is empty: {tmp_file_path}")
            
            print(f"[DeepSeek-OCR] Processing image: {tmp_file_path} ({file_size} bytes)", flush=True)
            sys.stdout.flush()
            
            # Create temporary output directory (model requires a valid path, not None)
            # The model's infer() method calls os.makedirs(output_path) even when save_results=False
            output_dir = tempfile.mkdtemp(prefix="deepseek_ocr_output_")
            print(f"[DeepSeek-OCR] Created output directory: {output_dir}", flush=True)
            sys.stdout.flush()
            
            # Run inference
            print(f"[DeepSeek-OCR] Calling model.infer()...", flush=True)
            sys.stdout.flush()
            result = model.infer(
                tokenizer,
                prompt=ocr_prompt,
                image_file=tmp_file_path,
                output_path=output_dir,  # Use temp directory instead of None
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=True,  # Set to True to get results returned
                test_compress=False
            )
            print(f"[DeepSeek-OCR] model.infer() completed, result type: {type(result)}", flush=True)
            print(f"[DeepSeek-OCR] model.infer() result value: {str(result)[:200] if result else 'None'}", flush=True)
            sys.stdout.flush()
            
            # Check if result is None - model might have written to file instead
            if result is None:
                print(f"[DeepSeek-OCR] Result is None, checking output directory for saved files...", flush=True)
                # Check if there are any text files in the output directory
                output_files = []
                if os.path.exists(output_dir):
                    output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
                    print(f"[DeepSeek-OCR] Found {len(output_files)} files in output directory: {output_files}", flush=True)
                    
                    # Try to read text from any .txt files
                    for filename in output_files:
                        if filename.endswith('.txt'):
                            filepath = os.path.join(output_dir, filename)
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    text = f.read()
                                    print(f"[DeepSeek-OCR] Read {len(text)} characters from {filename}", flush=True)
                                    sys.stdout.flush()
                                    return text
                            except Exception as e:
                                print(f"[DeepSeek-OCR] Failed to read {filename}: {e}", flush=True)
                
                # If no files found, return empty string or raise error
                raise ValueError("Model returned None and no output files were found in output directory")
            
            # Extract text from result
            text = result if isinstance(result, str) else str(result)
            print(f"[DeepSeek-OCR] Extracted text length: {len(text)}", flush=True)
            sys.stdout.flush()
            
            return text
        
        finally:
            # Clean up temp file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    print(f"[DeepSeek-OCR] Cleaned up temp file: {tmp_file_path}", flush=True)
                except Exception as e:
                    print(f"[DeepSeek-OCR] Warning: Failed to delete temp file {tmp_file_path}: {e}", flush=True)
            
            # Clean up output directory
            if output_dir and os.path.exists(output_dir):
                try:
                    import shutil
                    shutil.rmtree(output_dir)
                    print(f"[DeepSeek-OCR] Cleaned up output directory: {output_dir}", flush=True)
                except Exception as e:
                    print(f"[DeepSeek-OCR] Warning: Failed to delete output directory {output_dir}: {e}", flush=True)
    
    except Exception as e:
        print(f"[DeepSeek-OCR] ERROR in process_image(): {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

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
        print(f"[DeepSeek-OCR] Image data type: {type(image_data)}", flush=True)
        print(f"[DeepSeek-OCR] Image data length: {len(image_data) if image_data else 'None'}", flush=True)
        print(f"[DeepSeek-OCR] Image data preview (first 100 chars): {str(image_data)[:100] if image_data else 'None'}", flush=True)
        sys.stdout.flush()
        
        if not image_data:
            error_msg = "Missing 'image' field in input. Provide base64 encoded image, data URL, or image URL."
            print(f"[DeepSeek-OCR] ERROR: {error_msg}", flush=True)
            return {
                "error": error_msg
            }
        
        # Get mode (default: gundam)
        mode = input_data.get("mode", "gundam")
        if mode not in SIZE_MODES:
            mode = "gundam"
        print(f"[DeepSeek-OCR] Using mode: {mode}", flush=True)
        
        # Get format (text or markdown)
        format_type = input_data.get("format", "text")
        print(f"[DeepSeek-OCR] Format type: {format_type}", flush=True)
        
        # Determine prompt based on format
        prompt = input_data.get("prompt")
        if not prompt:
            if format_type == "markdown":
                prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            else:
                prompt = "<image>\nFree OCR. "
        print(f"[DeepSeek-OCR] Using prompt: {prompt[:50]}...", flush=True)
        sys.stdout.flush()
        
        # Process image
        print(f"[DeepSeek-OCR] Calling process_image()...", flush=True)
        sys.stdout.flush()
        result_text = process_image(image_data, mode=mode, prompt=prompt)
        print(f"[DeepSeek-OCR] process_image() completed, result length: {len(result_text) if result_text else 'None'}", flush=True)
        sys.stdout.flush()
        
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
        import traceback
        error_msg = str(e)
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()
        
        print(f"[DeepSeek-OCR] ========== HANDLER ERROR ==========", flush=True)
        print(f"[DeepSeek-OCR] Error type: {error_type}", flush=True)
        print(f"[DeepSeek-OCR] Error message: {error_msg}", flush=True)
        print(f"[DeepSeek-OCR] Traceback:", flush=True)
        print(error_traceback, flush=True)
        print(f"[DeepSeek-OCR] ====================================", flush=True)
        sys.stdout.flush()
        
        return {
            "error": error_msg,
            "type": error_type
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

