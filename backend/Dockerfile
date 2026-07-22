FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

# SAM2 source is baked into the image so Coolify never needs a manual clone step.
# The patch below disables torch.jit.script() on SAM2Transforms — it segfaults on
# some aarch64 PyTorch builds (TorchScript codegen bug); eager mode is correct,
# just marginally slower.
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /app/sam2 && \
    python3.11 -c "import pathlib; p = pathlib.Path('/app/sam2/sam2/utils/transforms.py'); s = p.read_text(); old = 'torch.jit.script(\n            nn.Sequential(\n                Resize((self.resolution, self.resolution)),\n                Normalize(self.mean, self.std),\n            )\n        )'; new = 'nn.Sequential(\n            Resize((self.resolution, self.resolution)),\n            Normalize(self.mean, self.std),\n        )'; assert old in s, 'pattern not found'; p.write_text(s.replace(old, new))" && \
    python3.11 -m pip install --no-cache-dir -e /app/sam2

COPY . .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
