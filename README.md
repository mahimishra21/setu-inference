---
title: SETU Sign Recognizer
emoji: 🤟
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SETU sign-recognition inference service

Wraps [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE)'s pretrained
transformer behind one HTTP endpoint, so the SETU kiosk (`apps/web`) can send
a citizen's signed clip and get a gloss back. Deliberately **not** on Vercel:
it needs PyTorch, MediaPipe, and a ~200MB checkpoint, none of which belong in
a serverless edge function. See `/CREDITS.md` at the repo root for full
attribution and license text.

- `POST /recognize` — multipart field `clip` (webm/mp4, ~1-2s) →
  `{gloss: string | null, confidence: number, topk: [{gloss, confidence}]}`.
  `gloss` is `null` when the top prediction is below the 0.60 confidence
  floor, or when no frames could be read from the clip at all.
- `GET /health` — `{status: "ok", model_loaded: boolean}`.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.*
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The checkpoint must already exist at
`vendor/INCLUDE/checkpoints/include_no_cnn_transformer_large.pth` (see
"The checkpoint" below) — the app raises a clear error at startup if it's
missing rather than silently serving nothing.

## Testing

```bash
pip install pytest
pytest test_app.py -v
```

These build synthetic video clips with OpenCV and post them through the real
HTTP endpoint — real MediaPipe extraction, real model, real multipart upload
handling. They prove the *pipeline* is wired correctly (a clip in gets a
well-formed prediction out, the confidence floor is applied, a corrupted
upload degrades to `gloss: null` instead of a 500). They do **not** and
cannot prove recognition *accuracy* — there is no real sign in a synthetic
clip. That number can only come from testing with actual signers, which is
why Milestone 3's report separates "the plumbing works" from "here is the
accuracy," same as Milestone 1 did.

Measured on this dev machine's CPU: ~0.85s per clip end to end (MediaPipe
extraction + model forward pass). Expect this to be somewhat higher on a
free-tier host's shared CPU.

## The checkpoint

`vendor/INCLUDE/checkpoints/include_no_cnn_transformer_large.pth` (205,805,173
bytes) is gitignored in both this repo and the main SETU repo — a 200MB
binary doesn't belong in either's git history. It's published as a
[GitHub Release asset](https://github.com/mahimishra21/setu-inference/releases/tag/checkpoint-v1)
on this repo instead (a plain HTTPS download, works on any host), and the
Dockerfile fetches and byte-count-verifies it during the build.

Git LFS was tried first and dropped: LFS support turned out to vary by build
host — a build that clones the repo without running the LFS smudge step
silently leaves a ~134-byte pointer file in the checkpoint's place, and
`torch.load()` then fails at startup with a confusing error that gives no
hint the actual problem is upstream, in the clone. A GitHub Release asset
sidesteps that entirely: it's the same URL and the same bytes regardless of
whether the host understands LFS.

Running locally without Docker: download that release asset yourself into
`vendor/INCLUDE/checkpoints/include_no_cnn_transformer_large.pth` before
starting `uvicorn` — the app raises a clear error at startup if it's missing.

## Deploying (Render, free tier)

Hugging Face Spaces' Docker SDK requires a paid PRO plan as of this writing
(only its Static SDK is free, which can't run Python) — that's why this
targets Render instead. Render deploys straight from a GitHub repo, so there's
no separate git-push-to-a-different-remote step.

Requires a free render.com account — that step has to happen in your own
browser.

1. Sign up at <https://render.com> (GitHub sign-in is easiest).
2. Dashboard → **New +** → **Web Service** → connect your GitHub account if
   asked → select the `setu-inference` repo.
3. Render should auto-detect the `Dockerfile`. Set **Instance Type** to
   **Free**.
4. **Settings → Environment** → add `ALLOWED_ORIGIN` = your Vercel production
   URL (e.g. `https://web-smoky-chi-27.vercel.app`), so the browser's CORS
   preflight succeeds. Leaving it unset defaults to `*`, which works but is
   wide open.
5. Create the service and watch the **Logs** tab (the torch/mediapipe install
   takes several minutes the first time). Once it says
   `Application startup complete`, it's live at
   `https://<service-name>.onrender.com`.
6. Confirm it: `curl https://<service-name>.onrender.com/health` should
   return `{"status":"ok","model_loaded":true}`.
7. Give that URL to Claude to wire into Vercel (`NEXT_PUBLIC_RECOGNIZER_URL`)
   — the Vercel CLI here is already authenticated to your account, so that
   part doesn't need you to do anything further.

Free-tier Render services spin down after 15 minutes idle and take about a
minute to wake back up on the next request — the first citizen after a quiet
spell will see a slow first clip. A paid instance removes that; not needed to
prove the pilot works.
