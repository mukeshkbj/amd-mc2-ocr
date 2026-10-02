# amd-mc2-ocr — Mini-Challenge 2 (Lablab × AMD AI Academy)

Optical character recognition for license plates and road signs, packaged as a
Docker image on the mandated ROCm PyTorch base. A resident
**Qwen3-VL-8B-Instruct** daemon loads once at container start; each
`app.py --input-image` exec is a thin client over a Unix socket, so per-image
inference completes in well under the 30-second budget.

Published image: `docker.io/mukeshkbj/amd-mc2-ocr:v1` (~34 GB, uncompressed).

## Layout

```
app/daemon.py   resident socket server — loads the model once, warmup included
app/ocr.py      engine: preprocess -> VLM generate -> transcription rules
app/app.py      grader entrypoint: --input-image -> /app/output/<stem>_output.json
tools/run_eval.py    grader-style scorer (normalized matching, timeout flags)
tools/make_harder.py degraded-variant generator (noise/blur/rotate/low-light)
Dockerfile      rocm/pytorch base + weights baked in (no network at eval)
```

## Transcription rules implemented

- US plates: jurisdiction banners and slogans omitted; the plate string only
- Chinese plates: province prefix preserved (e.g. `京A·12345`)
- Signs: all printed text, top-to-bottom; advisory plaques return digits only
- Empty/unreadable images return an empty text with low confidence — never prose

## Results

- 10/10 official sample images, 0.3–0.6 s per image (AMD GPU pod)
- 60/60 generated degraded variants (noise, blur, rotation, low light)
- Verified with `--network none` — the model is inside the image

## Build

```bash
# model weights are NOT in this repo — fetch them first (~17 GiB)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b

docker build -t amd-mc2-ocr:v1 .
docker run -d --name mc2 amd-mc2-ocr:v1          # GPU host (ROCm)
docker cp <images>/. mc2:/app/input/
docker exec mc2 python3 /app/app.py --input-image /app/input/x.png
docker exec mc2 cat /app/output/x_output.json    # {"text": "...", "confidence": ...}
```

Env knobs: `MC2_MODEL_DIR`, `MC2_MAX_NEW_TOKENS` (default 96), `MC2_SOCK`
(unix path, or `tcp:host:port` for non-Linux testing), `MC2_READY`.

No GPU fallback by design: without a visible GPU the daemon exits — a CPU run
would score zero and must not look like success.
