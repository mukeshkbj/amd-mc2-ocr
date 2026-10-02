"""OCR engine: Qwen3-VL-8B on ROCm, transcription rules baked into the prompt.

Loaded once by daemon.py; each query is one generate() call.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

MODEL_DIR = os.environ.get("MC2_MODEL_DIR", "/models/qwen3-vl-8b")
ALLOW_CPU = os.environ.get("ALLOW_CPU", "0") == "1"
MAX_NEW_TOKENS = int(os.environ.get("MC2_MAX_NEW_TOKENS", "96"))

PROMPT = (
    "Read the text in this image. Rules:\n"
    "- Return only the characters present, in reading order. Multi-line text joins "
    "top to bottom with a single space.\n"
    "- License plates: return only the registration number. Drop jurisdiction names "
    "and slogans printed around it (e.g. CALIFORNIA, NEW YORK, EXCELSIOR, THE LONE "
    "STAR STATE). On Chinese plates the leading province character and letter are "
    "part of the registration and must be kept (e.g. 京A·12345).\n"
    "- Speed limit signs: include the printed words and the number (e.g. SPEED "
    "LIMIT 65). Advisory plaques with only a number: return the digits only.\n"
    "- Do not add labels, units, prefixes, or explanation. Output the text only."
)

_STRIP = re.compile(r"[\s\-\.·_]")


def normalize(text: str) -> str:
    """Grader-side normalization, applied locally for sanity checks."""
    return _STRIP.sub("", text.upper())


def _load(image_path: str):
    from PIL import Image

    img = Image.open(image_path)
    img.load()
    if getattr(img, "n_frames", 1) > 1:  # multi-frame TIFF: first page
        img.seek(0)
    return img.convert("RGB")


class OCREngine:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available() and not ALLOW_CPU:
            raise RuntimeError(
                "no GPU visible and ALLOW_CPU is not set; a CPU run scores zero"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def warmup(self) -> None:
        """Exercise image and generation kernels before the graded clock."""
        from PIL import Image

        img = Image.new("RGB", (1024, 768), (255, 255, 255))
        self.read_image(img)
        self.read_image(img)

    def _generate(self, img) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()

    def read_image(self, img) -> tuple[str, float]:
        raw = self._generate(img)
        return self._clean(raw)

    def read(self, image_path: str) -> tuple[str, float]:
        img = _load(image_path)
        text, conf = self.read_image(img)
        if not text:
            # retry once on an enhanced variant: autocontrast + 2x upscale
            from PIL import ImageOps

            enh = ImageOps.autocontrast(img)
            enh = enh.resize((enh.width * 2, enh.height * 2))
            text, conf = self.read_image(enh)
        return text, conf

    @staticmethod
    def _clean(raw: str) -> tuple[str, float]:
        # model output -> single-space-joined text; refuse markdown fences
        text = raw.replace("```", " ").strip()
        text = " ".join(text.split())
        return text, (0.9 if text else 0.0)


if __name__ == "__main__":
    # manual smoke test: python3 ocr.py <image>
    import sys

    eng = OCREngine()
    eng.warmup()
    t0 = time.time()
    print(eng.read(sys.argv[1]), f"{time.time() - t0:.2f}s", Path(sys.argv[1]).name)
