FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV ONNXOCR_PORT=5005
ENV ONNXOCR_USE_GPU=1
ENV ONNXOCR_OCR_MODEL_SIZE=small
ENV ONNXOCR_ORT_LOG_SEVERITY=3
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN grep -v '^onnxruntime' requirements.txt > /tmp/requirements-gpu.txt \
    && pip install --no-cache-dir -r /tmp/requirements-gpu.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && pip install --no-cache-dir \
        onnxruntime-gpu==1.20.1 \
        nvidia-cudnn-cu12==9.23.2.1 \
        nvidia-cublas-cu12==12.9.2.10 \
        nvidia-cuda-nvrtc-cu12==12.9.86 \
        -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install --no-cache-dir \
        nvidia-curand-cu12==10.3.10.19 \
        nvidia-cufft-cu12==11.4.1.4 \
        nvidia-cuda-runtime-cu12==12.9.79 \
        -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY . .

EXPOSE 5005

CMD ["python", "app-service.py"]
