# MC2 — OCR on the mandated ROCm base. Do NOT squash/flatten: base-layer
# identity is checked before the container runs.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# --no-deps everywhere: a stray dep can silently replace ROCm torch with a
# CUDA build. Verified right after install so the build fails loudly.
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt && \
    python3 -c "import torch, transformers, huggingface_hub, httpx2; assert 'rocm' in torch.__version__, torch.__version__"

# Ship weights in the image: download them to ./models before building:
#   python3 -m pip install huggingface-hub
#   huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b
COPY models/ /models/
RUN test -s /models/qwen3-vl-8b/config.json
ENV MC2_MODEL_DIR=/models/qwen3-vl-8b
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY app/ /app/
RUN mkdir -p /app/input /app/output

# Model load happens at container start, inside the 10-minute startup budget;
# each exec'd app.py is then a thin socket client. Container must simply stay up.
CMD ["sh", "-c", "python3 /app/daemon.py >> /tmp/mc2-daemon.log 2>&1 & exec sleep infinity"]
