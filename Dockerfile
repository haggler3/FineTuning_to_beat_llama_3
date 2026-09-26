# CUDA 12.4 + cuDNN runtime, matching the transformers/bitsandbytes pins in
# requirements.txt. Build args let you point at a different base if your
# fleet standardizes on another CUDA minor version - the Python packages
# themselves are not CUDA-version-sensitive at this range.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Dependencies first, so an edit to src/ doesn't invalidate this layer.
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
RUN pip install --no-deps -e .

# Runs as a non-root user; HF_HOME is a mountable volume so model/dataset
# downloads survive across container restarts instead of re-fetching.
RUN useradd --create-home --uid 1000 runner
ENV HF_HOME=/home/runner/.cache/huggingface
RUN mkdir -p "$HF_HOME" && chown -R runner:runner /app "$HF_HOME"
USER runner

# No default CMD: this image is a base for qlora-train / qlora-eval / qlora-tokenize,
# run explicitly so the invocation (and its exit code) is visible in your
# orchestrator's logs rather than hidden behind an entrypoint script.
#
#   docker run --gpus all -e HUGGINGFACE_HUB_TOKEN -v hf-cache:/home/runner/.cache/huggingface \
#     qlora-icl qlora-train --model_id ... --torch_dataset_url ... --no_push
