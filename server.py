#!/usr/bin/env python3
"""
stem-demo server — upload a track, separate with micro models, serve stems.
"""

import http.server
import json
import os
import subprocess
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

PORT = 8090
ROOT = Path(__file__).parent
MODELS = ROOT / "models"
UPLOAD_DIR = ROOT / "uploads"
STEMS_DIR = ROOT / "stems"

UPLOAD_DIR.mkdir(exist_ok=True)
STEMS_DIR.mkdir(exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/separate":
            self.handle_separate()
        else:
            self.send_error(404)

    def handle_separate(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0 or content_length > 200 * 1024 * 1024:
            self.send_json({"error": "file too large or empty"}, 400)
            return

        # read upload
        data = self.rfile.read(content_length)

        # parse multipart manually — find the file
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "expected multipart/form-data"}, 400)
            return

        boundary = content_type.split("boundary=")[1].strip()
        parts = data.split(f"--{boundary}".encode())

        file_data = None
        filename = "upload.wav"
        for part in parts:
            if b"Content-Disposition" in part and b'name="file"' in part:
                # extract filename
                header_end = part.index(b"\r\n\r\n") + 4
                header = part[:header_end].decode(errors="replace")
                if 'filename="' in header:
                    fn = header.split('filename="')[1].split('"')[0]
                    if fn:
                        filename = fn
                file_data = part[header_end:]
                # strip trailing \r\n
                if file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]
                break

        if file_data is None:
            self.send_json({"error": "no file found in upload"}, 400)
            return

        # save to unique dir
        job_id = uuid.uuid4().hex[:8]
        job_dir = STEMS_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        input_path = UPLOAD_DIR / f"{job_id}_{filename}"
        input_path.write_bytes(file_data)

        # run inference
        inference_bin = ROOT / "inference"
        if not inference_bin.exists():
            self.send_json({"error": "inference binary not built — run make"}, 500)
            return

        cmd = [
            str(inference_bin),
            str(MODELS / "vocals.bin"),
            str(MODELS / "drums.bin"),
            str(MODELS / "bass.bin"),
            str(MODELS / "other.bin"),
            str(input_path),
            str(job_dir),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self.send_json({"error": f"separation failed: {result.stderr}"}, 500)
                return
        except subprocess.TimeoutExpired:
            self.send_json({"error": "separation timed out"}, 500)
            return

        # return stem URLs
        stems = {}
        for name in ["vocals", "drums", "bass", "other"]:
            stem_path = job_dir / f"{name}.wav"
            if stem_path.exists():
                stems[name] = f"/stems/{job_id}/{name}.wav"

        self.send_json({"stems": stems, "info": result.stdout.strip()})

        # clean up input file
        input_path.unlink(missing_ok=True)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.serve_file(ROOT / "index.html", "text/html")
        elif path.startswith("/stems/"):
            rel = path[1:]
            full = ROOT / rel
            if full.exists() and full.is_file():
                self.serve_file(full, "audio/wav")
            else:
                self.send_error(404)
        elif path.startswith("/models/") or path in ("/demo.mp3", "/stem.js", "/stem.wasm", "/worker.js"):
            full = ROOT / path.lstrip("/")
            if full.exists() and full.is_file():
                ct = {
                    ".js": "application/javascript", ".wasm": "application/wasm",
                    ".mp3": "audio/mpeg", ".bin": "application/octet-stream",
                }.get(full.suffix, "application/octet-stream")
                self.serve_file(full, ct)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def serve_file(self, path, content_type):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"  {args[0]}")


if __name__ == "__main__":
    print(f"stem-demo running at http://localhost:{PORT}")
    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
