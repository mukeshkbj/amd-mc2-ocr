#!/usr/bin/env python3
"""MC2 grader entrypoint. Thin client: each exec call talks to the resident
daemon over a unix socket, so the model loads once during container start
(the 10-minute startup budget) instead of once per image (the 30s budget).

    python3 /app/app.py --input-image /app/input/image_01.png [--output-dir /app/output]

Writes /app/output/<stem>_output.json -> {"text": "...", "confidence": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# default: unix socket (eval container). "tcp:host:port" for local/pod testing.
def _addr():
    raw = os.environ.get("MC2_SOCK", "/tmp/mc2-ocr.sock")
    if raw.startswith("tcp:"):
        host, port = raw[4:].rsplit(":", 1)
        return socket.AF_INET, (host, int(port))
    return socket.AF_UNIX, raw


READY_PATH = Path(os.environ.get("MC2_READY", "/tmp/mc2-ocr.ready"))
OUTPUT_DIR_DEFAULT = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
READY_TIMEOUT_S = float(os.environ.get("MC2_READY_TIMEOUT", "120"))


def _spawn_daemon() -> None:
    """Fallback if CMD did not bring the daemon up (e.g. manual testing)."""
    daemon = Path(__file__).resolve().parent / "daemon.py"
    subprocess.Popen(
        [sys.executable, str(daemon)],
        stdout=open("/tmp/mc2-daemon.log", "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_ready() -> None:
    deadline = time.time() + READY_TIMEOUT_S
    spawned = False
    while time.time() < deadline:
        if READY_PATH.exists():
            return
        if not spawned:
            _spawn_daemon()
            spawned = True
        time.sleep(0.5)
    raise TimeoutError("OCR daemon never became ready")


def _query(image_path: str) -> dict:
    family, addr = _addr()
    s = socket.socket(family, socket.SOCK_STREAM)
    s.connect(addr)
    s.sendall(json.dumps({"image": image_path}).encode("utf-8"))
    s.shutdown(socket.SHUT_WR)
    buf = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    args = ap.parse_args()

    _wait_ready()
    resp = _query(os.path.abspath(args.input_image))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input_image).stem
    out = args.output_dir / f"{stem}_output.json"
    # write whole at once: a file read mid-flush parses as invalid JSON
    out.write_text(
        json.dumps(
            {
                "text": str(resp.get("text", "")),
                "confidence": float(resp.get("confidence", 0.0)),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
