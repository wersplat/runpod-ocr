FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install FastAPI and web server
RUN python3 -m pip install fastapi uvicorn[standard] pydantic==2.8.2

# Install PyTorch with CUDA support
RUN python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# Install transformers and tokenizers
RUN python3 -m pip install transformers==4.46.3 tokenizers==0.20.3

# Install flash-attention (requires --no-build-isolation)
RUN python3 -m pip install flash-attn==2.7.3 --no-build-isolation

# Install additional dependencies
RUN python3 -m pip install einops addict easydict

# Install image processing libraries
RUN python3 -m pip install opencv-python-headless numpy pillow

WORKDIR /app
COPY app.py /app/app.py
EXPOSE 8000

# Default: FastAPI for local development
# For RunPod serverless, use Dockerfile.runpod instead
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]
