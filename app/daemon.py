"""Resident OCR server for MC2: load the model once, serve per-image exec calls.

Started by the container CMD. app.py connects over a unix socket; each request
is one JSON line {"image": "<abs path>"}, answered {"text": ..., "confidence": ...}.
The ready file exists only after the model is loaded and warmed, so clients can
poll for it instead of guessing.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import traceback
from pathlib import Path

# default: unix socket (eval container). "tcp:host:port" for local/pod testing
# on machines without AF_UNIX (e.g. Windows).
def _addr():
    raw = os.environ.get("MC2_SOCK", "/tmp/mc2-ocr.sock")
    if raw.startswith("tcp:"):
        host, port = raw[4:].rsplit(":", 1)
        return socket.AF_INET, (host, int(port)), None
    return socket.AF_UNIX, raw, raw


READY_PATH = Path(os.environ.get("MC2_READY", "/tmp/mc2-ocr.ready"))


def main() -> int:
    from ocr import OCREngine

    engine = OCREngine()
    engine.warmup()

    family, addr, sock_file = _addr()
    if sock_file and Path(sock_file).exists():
        Path(sock_file).unlink()
    srv = socket.socket(family, socket.SOCK_STREAM)
    srv.bind(addr)
    srv.listen(8)
    if sock_file:
        os.chmod(sock_file, 0o777)
    READY_PATH.touch()

    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                buf = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                req = json.loads(buf.decode("utf-8"))
                text, conf = engine.read(req["image"])
                resp = {"text": text, "confidence": conf}
            except Exception:
                traceback.print_exc()
                resp = {"text": "", "confidence": 0.0, "error": "inference failed"}
            conn.sendall(json.dumps(resp).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
