# SETU sign-recognition inference service.
#
# Deployed on Render (see README.md's "Deploying" section) -- Hugging Face
# Spaces' Docker SDK needs a paid plan as of this writing, and this has no
# other host-specific assumptions (PORT is read from the environment, falling
# back to 7860 for local `docker run`), so it isn't tied to Render either.

FROM python:3.12-slim

# libgl1/libglib2.0-0: mediapipe's compiled graph runtime dlopen's these even
# though opencv-python-headless itself doesn't need an X server. curl: fetches
# the checkpoint below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source so a code change doesn't invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.* \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY vendor/ vendor/

# The checkpoint (~205MB) is gitignored, not git-lfs -- LFS support turned
# out to vary by host (a build that clones without running the LFS smudge
# silently leaves a ~134-byte pointer file in its place, which then fails
# torch.load() at startup with no obvious cause). A plain HTTPS download from
# a GitHub Release asset works identically everywhere. The byte-count check
# turns "silently wrong file" into "build fails loudly," which is exactly the
# failure mode that cost real time to diagnose the first time.
RUN mkdir -p vendor/INCLUDE/checkpoints \
    && curl -fL -o vendor/INCLUDE/checkpoints/include_no_cnn_transformer_large.pth \
       https://github.com/mahimishra21/setu-inference/releases/download/checkpoint-v1/include_no_cnn_transformer_large.pth \
    && actual_size=$(stat -c%s vendor/INCLUDE/checkpoints/include_no_cnn_transformer_large.pth) \
    && [ "$actual_size" = "205805173" ] || (echo "checkpoint size mismatch: got $actual_size bytes" && exit 1)

ENV PYTHONUNBUFFERED=1
# Render sets PORT itself at runtime, overriding this default -- CMD below
# reads it dynamically either way.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
