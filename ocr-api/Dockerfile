FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip python3-dev git && rm -rf /var/lib/apt/lists/*

# Python deps
RUN python3 -m pip install --upgrade pip wheel setuptools
RUN python3 -m pip install fastapi uvicorn[standard] pydantic==2.8.2

# Paddle + OCR (GPU build)
RUN python3 -m pip install "paddlepaddle-gpu==2.6.1" -f https://www.paddlepaddle.org.cn/whl/linux/gpu/mkl/avx/stable.html
RUN python3 -m pip install paddleocr==2.7.0.3 opencv-python-headless numpy

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]