"""SETU sign-recognition inference service.

Wraps AI4Bharat/INCLUDE's pretrained transformer (vendor/INCLUDE) behind one
HTTP endpoint. Python-only, deliberately NOT deployed to Vercel — see
README.md for why and where this runs instead.

    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

VENDOR_DIR = Path(__file__).parent / "vendor" / "INCLUDE"
sys.path.insert(0, str(VENDOR_DIR))

# utils.load_label_map() reads "label_maps/label_map_include.json" -- a path
# relative to the process's working directory, not to the vendored package's
# own location. Without this the service only works when launched from
# inside vendor/INCLUDE, which is not a real deployment's working directory.
os.chdir(VENDOR_DIR)

# Deliberately imported only after the sys.path insert above -- these are the
# vendored (and SETU-patched, see vendor/INCLUDE/*.py PATCH comments)
# AI4Bharat modules, not a pip package.
from configs import TransformerConfig  # noqa: E402
from generate_keypoints import process_video  # noqa: E402
from models import Transformer  # noqa: E402
from utils import load_label_map  # noqa: E402

CHECKPOINT_PATH = VENDOR_DIR / "checkpoints" / "include_no_cnn_transformer_large.pth"
MAX_FRAME_LEN = 169  # matches vendor/INCLUDE/evaluate.py's KeypointsDataset

# Spec's confidence floor. Below this the app should ask the citizen to sign
# again rather than commit to a guess (Build Spec Sec 8.2's "low confidence").
CONFIDENCE_FLOOR = 0.60


class TopKEntry(BaseModel):
    gloss: str
    confidence: float


class RecognizeResponse(BaseModel):
    gloss: str | None
    confidence: float
    topk: list[TopKEntry]


class _State(TypedDict):
    model: Transformer
    label_map: dict[int, str]


_state: _State = {}  # type: ignore[typeddict-item]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Download it first -- see README.md."
        )

    config = TransformerConfig(size="large", max_position_embeddings=256)
    model = Transformer(config=config, n_classes=263)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    label_map = load_label_map("include")
    inv_label_map = {v: k for k, v in label_map.items()}

    _state["model"] = model
    _state["label_map"] = inv_label_map
    print(f"Model loaded. {len(inv_label_map)} classes. Checkpoint score: {ckpt.get('score')}")

    yield
    _state.clear()


app = FastAPI(title="SETU sign recognizer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # The citizen-facing kiosk is the only expected caller; tightened via
    # ALLOWED_ORIGIN at deploy time rather than left wide open in production
    # (see README.md's deploy section).
    allow_origins=[os.environ.get("ALLOWED_ORIGIN", "*")],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# evaluate.py's KeypointsDataset hardcodes these regardless of the source
# video's actual resolution (frame_length=1080, frame_width=1920 in its
# __init__, never overridden by evaluate.py's own call site). MediaPipe's
# coordinates are normalized to [0,1] relative to whatever frame size the
# video actually is, so the checkpoint's weights were trained on that [0,1]
# range multiplied by these two constants -- not on [0,1] itself, and not on
# the real pixel size of any given clip. This must be reproduced exactly:
# skipping it silently feeds the model inputs two to three orders of
# magnitude off from what it was trained on, with no error anywhere -- the
# forward pass runs fine and just returns confident, wrong answers.
_TRAIN_FRAME_WIDTH = 1920
_TRAIN_FRAME_HEIGHT = 1080


def _build_model_input(keypoints: dict) -> torch.Tensor:
    """Reproduces evaluate.py's KeypointsDataset formatting for one clip.

    Not reusing KeypointsDataset directly -- it's built around a directory of
    saved JSON files (glob + Dataset/__getitem__), which is the right shape
    for batch evaluation but awkward for one live request. Same math, though:
    pose(25x2) + hand1(21x2) + hand2(21x2) = 134 features/frame, linearly
    interpolated to fill gaps, rescaled to the training coordinate space,
    padded to MAX_FRAME_LEN.
    """
    import numpy as np
    import pandas as pd

    def interpolate(arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr)
        filled = df.interpolate(method="linear", limit_direction="both").to_numpy()
        if np.count_nonzero(~np.isnan(filled)) == 0:
            filled = np.zeros(filled.shape)
        return filled

    def combine(x_key: str, y_key: str, width: int, scale_x: float, scale_y: float) -> np.ndarray:
        x = np.array(keypoints[x_key], dtype=np.float32)
        y = np.array(keypoints[y_key], dtype=np.float32)
        if x.size == 0:
            n_frames = keypoints["n_frames"] or 1
            return np.zeros((n_frames, width), dtype=np.float32)
        x = interpolate(x) * scale_x
        y = interpolate(y) * scale_y
        # Interleave x/y the same way evaluate.py's combine_xy + the later
        # reshape(-1, width) does: pairs of columns, not x-block-then-y-block.
        out = np.empty((x.shape[0], width), dtype=np.float32)
        out[:, 0::2] = x
        out[:, 1::2] = y
        return out

    pose = combine("pose_x", "pose_y", 50, _TRAIN_FRAME_WIDTH, _TRAIN_FRAME_HEIGHT)
    h1 = combine("hand1_x", "hand1_y", 42, _TRAIN_FRAME_WIDTH, _TRAIN_FRAME_HEIGHT)
    h2 = combine("hand2_x", "hand2_y", 42, _TRAIN_FRAME_WIDTH, _TRAIN_FRAME_HEIGHT)

    final = np.concatenate([pose, h1, h2], axis=-1)
    final = np.nan_to_num(final, nan=0.0)

    if final.shape[0] >= MAX_FRAME_LEN:
        final = final[:MAX_FRAME_LEN]
    else:
        final = np.pad(final, ((0, MAX_FRAME_LEN - final.shape[0]), (0, 0)), "constant")

    return torch.FloatTensor(final).unsqueeze(0)  # (1, MAX_FRAME_LEN, 134)


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(clip: UploadFile = File(...)) -> RecognizeResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    suffix = Path(clip.filename or "clip.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory() as tmpdir:
        # process_video() (vendor/INCLUDE) reads the gloss label from the
        # parent directory name -- there is no real label for a live capture,
        # so this just needs to be a valid, harmless directory name.
        video_dir = Path(tmpdir) / "capture"
        video_dir.mkdir()
        video_path = video_dir / f"clip{suffix}"
        with open(video_path, "wb") as f:
            shutil.copyfileobj(clip.file, f)

        keypoints_dir = Path(tmpdir) / "keypoints"
        keypoints_dir.mkdir()
        try:
            process_video(str(video_path), str(keypoints_dir))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Could not process clip: {exc}") from exc

        keypoint_files = list(keypoints_dir.glob("*.json"))
        if not keypoint_files:
            raise HTTPException(status_code=422, detail="No frames could be read from the clip.")

        import json

        with open(keypoint_files[0]) as f:
            keypoints = json.load(f)

    if keypoints.get("n_frames", 0) == 0:
        return RecognizeResponse(gloss=None, confidence=0.0, topk=[])

    x = _build_model_input(keypoints)
    with torch.no_grad():
        logits = _state["model"](x)
    probs = torch.softmax(logits, dim=-1)[0]

    top5 = torch.topk(probs, k=min(5, probs.shape[0]))
    label_map = _state["label_map"]
    topk = [
        TopKEntry(gloss=label_map[idx].upper(), confidence=round(conf, 4))
        for conf, idx in zip(top5.values.tolist(), top5.indices.tolist())
    ]

    top_confidence = topk[0].confidence
    if top_confidence < CONFIDENCE_FLOOR:
        return RecognizeResponse(gloss=None, confidence=top_confidence, topk=topk)

    return RecognizeResponse(gloss=topk[0].gloss, confidence=top_confidence, topk=topk)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}
