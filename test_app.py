"""Integration tests for the FastAPI service (Milestone 3).

Runs the real app -- the real model checkpoint, the real vendored
generate_keypoints.py pipeline -- against synthetic video clips built with
OpenCV. Synthetic clips can't prove recognition *accuracy* (no real sign is in
them), but they exercise the exact code path a citizen's browser hits: a raw
multipart video upload in, a parsed JSON prediction out. See README.md for
where the honest accuracy number comes from instead (a real-signer test).

    pip install pytest
    pytest test_app.py -v
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app


def _make_clip(frames: int = 30, width: int = 320, height: int = 240) -> bytes:
    """A short synthetic mp4: two moving blobs, roughly hand-sized and colored."""
    path = "test_app_tmp_clip.mp4"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter for mp4v"
    for i in range(frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (30 + i * 6, 120 + int(40 * np.sin(i / 3))), 25, (200, 180, 160), -1)
        cv2.circle(frame, (width - 30 - i * 4, 100), 25, (200, 180, 160), -1)
        writer.write(frame)
    writer.release()

    with open(path, "rb") as f:
        data = f.read()
    import os

    os.remove(path)
    return data


@pytest.fixture(scope="module")
def client():
    # Entering the context manager runs the app's lifespan -- loads the real
    # checkpoint once for every test in this module, not once per test.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def synthetic_clip() -> bytes:
    return _make_clip()


class TestHealth:
    def test_reports_model_loaded(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


class TestRecognize:
    def test_accepts_a_real_clip_and_returns_the_declared_shape(
        self, client: TestClient, synthetic_clip: bytes
    ) -> None:
        response = client.post(
            "/recognize", files={"clip": ("clip.mp4", io.BytesIO(synthetic_clip), "video/mp4")}
        )
        assert response.status_code == 200
        body = response.json()

        assert "gloss" in body and (body["gloss"] is None or isinstance(body["gloss"], str))
        assert isinstance(body["confidence"], float)
        assert isinstance(body["topk"], list)
        for entry in body["topk"]:
            assert isinstance(entry["gloss"], str)
            assert isinstance(entry["confidence"], float)
        # topk is sorted best-first; confidence must be the winner's score.
        if body["topk"]:
            assert body["confidence"] == pytest.approx(body["topk"][0]["confidence"])

    def test_applies_the_060_confidence_floor(
        self, client: TestClient, synthetic_clip: bytes
    ) -> None:
        # Two bouncing circles are not a real ISL sign -- the model should
        # never be confident about one, which is exactly the case the floor
        # exists for (Build Spec Sec 8.2's "low confidence -> ask again").
        response = client.post(
            "/recognize", files={"clip": ("clip.mp4", io.BytesIO(synthetic_clip), "video/mp4")}
        )
        body = response.json()
        if body["confidence"] < 0.60:
            assert body["gloss"] is None

    def test_degrades_gracefully_on_a_file_that_isnt_a_video(self, client: TestClient) -> None:
        response = client.post(
            "/recognize",
            files={"clip": ("clip.mp4", io.BytesIO(b"not a real video, just bytes"), "video/mp4")},
        )
        # app.py deliberately treats zero readable frames as "no signal," not
        # an error: a corrupted/truncated recording (a dropped browser frame,
        # a flaky upload) should read to the citizen as "please sign again,"
        # the same as any other low-confidence result -- not a scary failure.
        assert response.status_code == 200
        body = response.json()
        assert body == {"gloss": None, "confidence": 0.0, "topk": []}

    def test_requires_the_clip_field(self, client: TestClient) -> None:
        response = client.post("/recognize")
        assert response.status_code == 422

    def test_latency_is_reasonable_on_cpu(self, client: TestClient, synthetic_clip: bytes) -> None:
        import time

        start = time.monotonic()
        response = client.post(
            "/recognize", files={"clip": ("clip.mp4", io.BytesIO(synthetic_clip), "video/mp4")}
        )
        elapsed = time.monotonic() - start
        assert response.status_code == 200
        # Generous CPU budget -- this is one clip, not a batch, and free-tier
        # hosts (Milestone 3) are slower than this dev machine.
        assert elapsed < 10.0, f"one clip took {elapsed:.1f}s on CPU"
