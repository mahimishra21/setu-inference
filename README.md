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
bytes) is gitignored in the main SETU repo — a 200MB binary doesn't belong in
a Next.js monorepo's git history. It is **not** re-downloaded during the
Docker build either, because there is no verified, stable URL for it recorded
anywhere in this repo to build that step against, and silently trusting an
unpinned "pretrained checkpoint" URL at build time is exactly the kind of
supply-chain risk not worth taking. Instead, get the already-verified file
(byte count, PyTorch zip signature, and a successful load against this exact
model architecture with zero `state_dict` mismatches were all checked when it
was first downloaded) into wherever you deploy this by one of:

- **Hugging Face Spaces**: push it with `git lfs` (see "Deploying" below), or
  drag it directly into the Space's Files tab in the browser — both work.
- **Any other host**: copy the file alongside `app.py` before building, or
  mount/download it at container start using your own trusted source.

## Deploying (Hugging Face Spaces, free CPU tier)

Requires a free huggingface.co account — that step, and logging in with your
own token, has to happen in your own browser/terminal.

1. Go to <https://huggingface.co/new-space>. Pick a name, set **SDK** to
   **Docker**, hardware **CPU basic** (free), visibility however you like.
   This gives you a git URL:
   `https://huggingface.co/spaces/<your-username>/<space-name>`.
2. In a terminal, from the repo root:

   ```bash
   cp -r inference-service /tmp/setu-space
   cd /tmp/setu-space
   git init
   git lfs install
   git lfs track "*.pth"
   git add -A
   git add -f vendor/INCLUDE/checkpoints/include_no_cnn_transformer_large.pth
   git commit -m "Deploy SETU sign recognizer"
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```

   (No `git lfs`? Skip the `lfs` lines, push everything else, then drag
   `include_no_cnn_transformer_large.pth` into the Space's **Files** tab in
   the browser afterward — same end result.)

3. In the Space's **Settings → Variables and secrets**, add
   `ALLOWED_ORIGIN` = your Vercel production URL (e.g.
   `https://web-smoky-chi-27.vercel.app`) so the browser's CORS preflight
   succeeds. Leaving it unset defaults to `*`, which works but is wide open.
4. Wait for the build to finish (the **Logs** tab shows progress — expect
   several minutes for the torch/mediapipe install). Once it says
   `Application startup complete`, the service is live at
   `https://<your-username>-<space-name>.hf.space`.
5. Confirm it: `curl https://<your-username>-<space-name>.hf.space/health`
   should return `{"status":"ok","model_loaded":true}`.
6. Give that URL to Claude to wire into Vercel (`NEXT_PUBLIC_RECOGNIZER_URL`)
   — the Vercel CLI here is already authenticated to your account, so that
   part doesn't need you to do anything further.

Free CPU-basic Spaces sleep after a period of inactivity and take a beat to
wake back up on the next request — the first citizen after a quiet spell will
see a slower first clip. Upgrading to a persistent (paid) CPU tier removes
that; not needed to prove the pilot works.
