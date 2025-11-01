# RunPod Serverless Deployment Guide

This guide explains how to deploy the DeepSeek-OCR API as a serverless endpoint on RunPod.

## Overview

RunPod serverless endpoints allow you to deploy containerized applications that scale automatically based on demand. The handler receives requests and returns responses in JSON format.

## Prerequisites

1. RunPod account with credits
2. Docker (for building the image locally, optional)
3. GitHub account (for container registry, or use RunPod's registry)

## Deployment Steps

### Option 1: Deploy via RunPod Console (Recommended)

1. **Build Docker Image**

   Build the image using the RunPod-specific Dockerfile:
   ```bash
   cd ocr-api
   docker build -f Dockerfile.runpod -t deepseek-ocr:latest .
   ```

2. **Push to Container Registry**

   Tag and push to Docker Hub, GitHub Container Registry, or RunPod's registry:
   ```bash
   # Docker Hub example
   docker tag deepseek-ocr:latest yourusername/deepseek-ocr:latest
   docker push yourusername/deepseek-ocr:latest
   ```

3. **Create Serverless Endpoint in RunPod**

   - Go to RunPod Dashboard → Serverless → Create Endpoint
   - Container image: `yourusername/deepseek-ocr:latest`
   - GPU: Select GPU type (recommended: RTX 4090 or A100)
   - Container disk: 20GB+ (model is ~6GB)
   - Max workers: 1-5 (depending on your needs)
   - FlashBoot: Enable (for faster cold starts)

4. **Configure Environment Variables** (optional)
   - `DEEPSEEK_OCR_BASE_SIZE`: 1024
   - `DEEPSEEK_OCR_IMAGE_SIZE`: 640
   - `DEEPSEEK_OCR_CROP_MODE`: true

### Option 2: Deploy via RunPod API

```python
import runpod

# Configure your API key
runpod.api_key = "your-api-key"

# Create serverless endpoint
endpoint = runpod.create_endpoint(
    name="deepseek-ocr",
    image="yourusername/deepseek-ocr:latest",
    gpu_type_id="NVIDIA RTX 4090",  # or your preferred GPU
    container_disk_in_gb=20,
    max_workers=1,
    flashboot=True
)

print(f"Endpoint created: {endpoint['id']}")
```

## Usage

### Python Client

```python
import runpod
import base64

# Load your API key
runpod.api_key = "your-runpod-api-key"

# Load image
with open("document.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Call endpoint
endpoint_id = "your-endpoint-id"
job = runpod.serverless.run(
    endpoint_id=endpoint_id,
    input={
        "image": image_base64,
        "mode": "gundam",
        "format": "text"  # or "markdown"
    }
)

# Get result
result = job.output()
print(result["text"])
```

### cURL Example

```bash
curl -X POST \
  https://api.runpod.io/v2/your-endpoint-id/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "image": "base64_encoded_image_here",
      "mode": "gundam",
      "format": "text"
    }
  }'
```

### JavaScript/TypeScript Example

```javascript
const runpod = require('@runpod/serverless-sdk');

const endpointId = 'your-endpoint-id';
const imageBase64 = '...'; // base64 encoded image

async function runOCR() {
  const result = await runpod.run(endpointId, {
    input: {
      image: imageBase64,
      mode: 'gundam',
      format: 'text'
    }
  });
  
  console.log(result.output.text);
}
```

## Input Format

The handler expects the following input format:

```json
{
  "input": {
    "image": "<base64_encoded_image> or <image_url>",
    "mode": "gundam",  // optional: tiny, small, base, large, gundam
    "prompt": "<custom_prompt>",  // optional
    "format": "text"  // optional: "text" or "markdown"
  }
}
```

**Fields:**
- `image` (required): Base64-encoded image string or HTTP/HTTPS URL
- `mode` (optional): Size mode preset (tiny, small, base, large, gundam). Default: `gundam`
- `prompt` (optional): Custom prompt for OCR. Default: `"<image>\nFree OCR. "` or markdown prompt
- `format` (optional): `"text"` or `"markdown"`. Default: `"text"`

## Output Format

**Text Format:**
```json
{
  "text": "Extracted text content...",
  "mode": "gundam"
}
```

**Markdown Format:**
```json
{
  "markdown": "# Document Title\n\nContent...",
  "mode": "gundam"
}
```

**Error Format:**
```json
{
  "error": "Error message here",
  "type": "ErrorType"
}
```

## Size Modes

- **tiny**: Fastest, smallest memory (512x512)
- **small**: Balanced speed/quality (640x640)
- **base**: Higher quality (1024x1024)
- **large**: Highest quality, slower (1280x1280)
- **gundam**: Optimized for documents (1024 base, 640 image, crop enabled)

## Cost Optimization

1. **Enable FlashBoot**: Reduces cold start time
2. **Use appropriate GPU**: RTX 4090 is often sufficient
3. **Monitor workers**: Adjust max_workers based on traffic
4. **Use smaller modes**: For faster/cheaper processing when quality allows

## Troubleshooting

### Model Loading Issues

If you encounter OOM errors:
- Reduce `max_workers` to 1
- Use a larger GPU (A100 40GB or higher)
- Try smaller size modes (tiny/small)

### Timeout Issues

- Increase timeout in RunPod endpoint settings
- Use smaller size modes for faster processing
- Enable FlashBoot to keep model loaded

### Cold Start Delays

- Enable FlashBoot to keep at least one worker warm
- Model loading takes ~30-60 seconds on first request

## Monitoring

Monitor your endpoint in RunPod dashboard:
- Request count
- Average response time
- Error rate
- GPU utilization
- Cost per request

## Pricing

RunPod charges based on:
- GPU time used
- Cold start time
- Number of workers

Typical cost: $0.01-0.05 per OCR request depending on image size and GPU type.

## Example: Complete Workflow

```python
import runpod
import base64
from PIL import Image
import io

# Initialize
runpod.api_key = "your-api-key"
endpoint_id = "your-endpoint-id"

# Prepare image
img = Image.open("document.jpg")
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
img_base64 = base64.b64encode(buffer.getvalue()).decode()

# Run OCR
job = runpod.serverless.run(
    endpoint_id=endpoint_id,
    input={
        "image": img_base64,
        "mode": "gundam",
        "format": "markdown"
    }
)

# Wait for result
result = job.output(timeout=300)  # 5 minute timeout

if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(result["markdown"])
```

## Additional Resources

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [DeepSeek-OCR Model Card](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

