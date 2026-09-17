# SETU sign-recognition inference service.
#
# Built for Hugging Face Spaces' Docker SDK (see README.md's YAML header --
# app_port must match the port this container listens on). Nothing here is
# HF-specific beyond that port and the working directory; it also runs fine
# with a plain `docker run -p 8000:8000 ...` for local testing.

FROM python:3.12-slim

# libgl1/libglib2.0-0: mediapipe's compiled graph runtime dlopen's these even
# though opencv-python-headless itself doesn't need an X server.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source so a code change doesn't invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.* \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY vendor/ vendor/

# The checkpoint (vendor/INCLUDE/checkpoints/*.pth, ~205MB) is gitignored in
# the main SETU repo and is NOT fetched here -- see README.md's "Deploying"
# section for why (no stable, verified re-download URL) and how to get it
# into this image (push it via git-lfs, or drag it into the Space's Files
# tab). The service raises a clear startup error if it's missing rather than
# silently serving a model that was never loaded.

ENV PYTHONUNBUFFERED=1
# HF Spaces' Docker SDK routes traffic to this port; see the app_port in
# README.md's YAML header. Override PORT for a non-Spaces deployment.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
