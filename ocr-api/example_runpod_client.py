# Example: Using DeepSeek-OCR RunPod Serverless Endpoint

import runpod
import base64
from pathlib import Path

# Set your RunPod API key
runpod.api_key = "YOUR_RUNPOD_API_KEY"

# Your endpoint ID (get this from RunPod dashboard)
ENDPOINT_ID = "your-endpoint-id-here"

def ocr_image(image_path: str, mode: str = "gundam", format_type: str = "text"):
    """
    Process an image with OCR using RunPod serverless endpoint.
    
    Args:
        image_path: Path to image file
        mode: Size mode (tiny, small, base, large, gundam)
        format_type: "text" or "markdown"
    
    Returns:
        dict with extracted text/markdown
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # Submit job
    job = runpod.serverless.run(
        endpoint_id=ENDPOINT_ID,
        input={
            "image": image_base64,
            "mode": mode,
            "format": format_type
        }
    )
    
    # Wait for result (with timeout)
    result = job.output(timeout=300)  # 5 minute timeout
    
    return result

def ocr_image_url(image_url: str, mode: str = "gundam", format_type: str = "text"):
    """
    Process an image from URL with OCR.
    
    Args:
        image_url: HTTP/HTTPS URL to image
        mode: Size mode
        format_type: "text" or "markdown"
    
    Returns:
        dict with extracted text/markdown
    """
    job = runpod.serverless.run(
        endpoint_id=ENDPOINT_ID,
        input={
            "image": image_url,  # Can pass URL directly
            "mode": mode,
            "format": format_type
        }
    )
    
    result = job.output(timeout=300)
    return result

# Example usage
if __name__ == "__main__":
    # Example 1: Process local image
    result = ocr_image("document.jpg", mode="gundam", format_type="text")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Extracted text: {result['text']}")
    
    # Example 2: Process image to markdown
    result = ocr_image("document.jpg", mode="base", format_type="markdown")
    if "error" not in result:
        print(f"Markdown:\n{result['markdown']}")
    
    # Example 3: Process from URL
    result = ocr_image_url(
        "https://example.com/document.jpg",
        mode="gundam",
        format_type="text"
    )
    print(result)

