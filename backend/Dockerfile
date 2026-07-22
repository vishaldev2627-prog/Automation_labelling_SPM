FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Install GPU-enabled PyTorch first, from PyTorch's own CUDA wheel index, before
# requirements.txt is installed — that way ultralytics/SAM2 see torch already
# satisfied and don't pull in a CPU-only default from PyPI (PyPI has no CUDA
# wheel for aarch64 at all). This host is aarch64 Grace Blackwell (GB10,
# compute capability 12.1 / sm_121); only cu128+ wheels carry Blackwell kernels.
# cu128 wheels bundle their own CUDA/cuDNN runtime via pip (nvidia-cudnn-cu12
# etc.), so the base image only needs to supply the driver-facing CUDA runtime,
# not a matching cuDNN — hence "runtime" with no cudnn suffix above.
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir \
        torch==2.7.1+cu128 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# SAM2 source is baked into the image so Coolify never needs a manual clone step.
# The patch below disables torch.jit.script() on SAM2Transforms — it segfaults on
# some aarch64 PyTorch builds (TorchScript codegen bug); eager mode is correct,
# just marginally slower.
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /app/sam2 && \
    python3.11 -c "import pathlib; p = pathlib.Path('/app/sam2/sam2/utils/transforms.py'); s = p.read_text(); old = 'torch.jit.script(\n            nn.Sequential(\n                Resize((self.resolution, self.resolution)),\n                Normalize(self.mean, self.std),\n            )\n        )'; new = 'nn.Sequential(\n            Resize((self.resolution, self.resolution)),\n            Normalize(self.mean, self.std),\n        )'; assert old in s, 'pattern not found'; p.write_text(s.replace(old, new))" && \
    python3.11 -m pip install --no-cache-dir -e /app/sam2

# Pre-download the MobileNetV3 ImageNet weights used for annotation
# propagation's similarity index, so the container doesn't need outbound
# internet on first use.
RUN python3.11 -c "import torchvision; torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)"

COPY . .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
