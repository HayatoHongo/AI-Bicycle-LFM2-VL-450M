# ============================================================
# Dockerfile for LFM2-VL Japanese Fine-Tuning
# CUDA 12.6 / PyTorch 2.8.0 / Transformers 4.55.0
# ============================================================

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# --- System Dependencies ---
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git wget curl vim build-essential ca-certificates \
    libjpeg-dev libpng-dev libgl1-mesa-glx && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- Install PyTorch (CUDA 12.6 build) ---
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126

# --- Copy and Install Remaining Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Environment Variables ---
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers
ENV TORCH_HOME=/workspace/.cache/torch

CMD ["bash"]