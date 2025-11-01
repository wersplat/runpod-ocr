# DeepSeek-OCR API

A FastAPI-based OCR service powered by [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR), a 3B parameter vision-language model optimized for document OCR with markdown conversion support.

## Features

- **High-quality OCR**: State-of-the-art text extraction from images and documents
- **Markdown Conversion**: Convert documents directly to markdown format
- **Multiple Size Modes**: Optimize for different use cases (Tiny, Small, Base, Large, Gundam)
- **GPU Accelerated**: Fast inference with CUDA support
- **Docker Ready**: Easy deployment with Docker Compose

## Requirements

- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- NVIDIA Container Toolkit

## Quick Start

1. Clone the repository:

```bash
git clone <repository-url>
cd runpod-ocr
```

2. Create a `.env` file (see `.env.example`):

```bash
cp .env.example .env
```

3. Start the service:

```bash
docker compose up -d
```

4. Test the API:

```bash
curl -X POST "http://localhost:8000/ocr/free" \
  -F "file=@your_image.jpg" \
  -F "mode=gundam"
```

## API Endpoints

### Health Check

```bash
GET /healthz
```

Returns `{"ok": true}` if the service is running.

### Free OCR

Extract plain text from an image:

```bash
POST /ocr/free
```

**Parameters:**

- `file` (multipart/form-data): Image file to process
- `mode` (query, optional): Size mode preset (tiny, small, base, large, gundam). Default: `gundam`

**Example:**

```bash
curl -X POST "http://localhost:8000/ocr/free?mode=gundam" \
  -F "file=@document.jpg"
```

**Response:**

```json
{
  "text": "Extracted text content...",
  "mode": "gundam"
}
```

### Markdown Conversion

Convert document to markdown format:

```bash
POST /ocr/markdown
```

**Parameters:**

- `file` (multipart/form-data): Image file to process
- `mode` (query, optional): Size mode preset. Default: `gundam`

**Example:**

```bash
curl -X POST "http://localhost:8000/ocr/markdown?mode=base" \
  -F "file=@document.jpg"
```

**Response:**

```json
{
  "markdown": "# Document Title\n\nParagraph content...",
  "mode": "base"
}
```

### Configurable OCR

Custom OCR with configurable prompt:

```bash
POST /ocr
```

**Parameters:**

- `file` (multipart/form-data): Image file to process
- `mode` (query, optional): Size mode preset. Default: `gundam`
- `prompt` (query, optional): Custom prompt. Default: `"<image>\nFree OCR. "`

**Example:**

```bash
curl -X POST "http://localhost:8000/ocr?mode=large&prompt=<image>%0AExtract%20all%20text" \
  -F "file=@document.jpg"
```

**Response:**

```json
{
  "text": "Extracted text...",
  "mode": "large"
}
```

## Size Modes

Choose the appropriate size mode based on your needs:

- **tiny**: `base_size=512, image_size=512, crop_mode=False` - Fastest, smallest memory footprint
- **small**: `base_size=640, image_size=640, crop_mode=False` - Balanced speed/quality
- **base**: `base_size=1024, image_size=1024, crop_mode=False` - Higher quality
- **large**: `base_size=1280, image_size=1280, crop_mode=False` - Highest quality, slower
- **gundam** (default): `base_size=1024, image_size=640, crop_mode=True` - Optimized for documents

## Configuration

Environment variables (set in `.env` file):

```bash
# DeepSeek-OCR Configuration
DEEPSEEK_OCR_BASE_SIZE=1024      # Base size for processing
DEEPSEEK_OCR_IMAGE_SIZE=640      # Image size for processing
DEEPSEEK_OCR_CROP_MODE=true      # Enable crop mode (true/false)

# GPU Configuration
CUDA_VISIBLE_DEVICES=0           # GPU device ID

# Service Configuration
OCR_PORT=8000                     # API port
```

## Python Client Example

```python
import requests

# Simple OCR
with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ocr/free',
        files={'file': f},
        params={'mode': 'gundam'}
    )
    result = response.json()
    print(result['text'])

# Markdown conversion
with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ocr/markdown',
        files={'file': f},
        params={'mode': 'base'}
    )
    result = response.json()
    print(result['markdown'])
```

## Migration from PaddleOCR

This project was migrated from PaddleOCR to DeepSeek-OCR. Key changes:

### API Changes

**Before (PaddleOCR):**

```python
POST /ocr
Response: {
  "lines": ["text1", "text2"],
  "boxes": [[[x1,y1], [x2,y2], ...], ...],
  "confidences": [0.95, 0.87]
}
```

**After (DeepSeek-OCR):**

```python
POST /ocr/free
Response: {
  "text": "Complete extracted text",
  "mode": "gundam"
}
```

### Model Differences

- **PaddleOCR**: Provided bounding boxes and per-word confidence scores
- **DeepSeek-OCR**: Provides complete text extraction with context awareness and markdown conversion support
- **Training**: PaddleOCR training scripts are preserved but deprecated. See `scripts/README-train.md` for details.

### Performance

- **DeepSeek-OCR**: 3B parameter model, optimized for document understanding
- **GPU Memory**: Requires ~8GB+ VRAM for optimal performance
- **Speed**: Processing time depends on size mode and image resolution

## Architecture

```
runpod-ocr/
├── ocr-api/
│   ├── app.py              # FastAPI application (local)
│   ├── handler.py          # RunPod serverless handler
│   ├── Dockerfile           # Container for local FastAPI
│   ├── Dockerfile.runpod    # Container for RunPod serverless
│   └── example_runpod_client.py  # Example RunPod client
├── docker-compose.yml      # Service orchestration
├── RUNPOD_DEPLOYMENT.md    # RunPod deployment guide
├── scripts/
│   ├── README-train.md     # Training documentation
│   └── train_rec.sh        # Legacy training script (deprecated)
├── models/                 # Model storage (optional custom models)
├── data/                   # Data directory
└── logs/                   # Application logs
```

## Development

### Building the Image

```bash
docker compose build ocr-api
```

### Viewing Logs

```bash
docker compose logs -f ocr-api
```

### Running in Development

```bash
docker compose up ocr-api
```

## RunPod Serverless Deployment

Deploy this API as a serverless endpoint on RunPod for automatic scaling and pay-per-use pricing.

### Quick Start

1. **Build the RunPod image:**
   ```bash
   cd ocr-api
   docker build -f Dockerfile.runpod -t yourusername/deepseek-ocr:latest .
   docker push yourusername/deepseek-ocr:latest
   ```

2. **Create endpoint in RunPod:**
   - Go to RunPod Dashboard → Serverless → Create Endpoint
   - Container: `yourusername/deepseek-ocr:latest`
   - GPU: RTX 4090 or A100 (recommended)
   - Container disk: 20GB+
   - Enable FlashBoot for faster cold starts

3. **Use the endpoint:**
   ```python
   import runpod
   import base64
   
   runpod.api_key = "your-api-key"
   
   with open("image.jpg", "rb") as f:
       image_b64 = base64.b64encode(f.read()).decode()
   
   job = runpod.serverless.run(
       endpoint_id="your-endpoint-id",
       input={
           "image": image_b64,
           "mode": "gundam",
           "format": "text"  # or "markdown"
       }
   )
   
   result = job.output()
   print(result["text"])
   ```

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for detailed deployment instructions and examples.

## References

- [DeepSeek-OCR on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Research Paper](https://arxiv.org/abs/2510.18234)
- [RunPod Serverless Documentation](https://docs.runpod.io/serverless)

## License

This project uses DeepSeek-OCR, which is licensed under MIT License. See the model card for details.
