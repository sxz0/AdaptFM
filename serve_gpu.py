#!/usr/bin/env python3
"""
AdaptFM GPU inference server.
Pattern: vLLM OpenAI server (subprocess, port 8081) + thin proxy on port 8080.
"""
import http.client
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

MODEL_DIR         = "/opt/ml/model"
PORT              = 8080
VLLM_PORT         = 8081
SERVED_MODEL_NAME = "Qwen/Qwen3.5-4B"
vllm_ready        = False


def start_vllm(model_path):
    max_model_len = os.environ.get("MAX_MODEL_LEN", "32768")
    gpu_mem_util  = os.environ.get("GPU_MEM_UTIL", "0.92")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_path,
        "--served-model-name",      SERVED_MODEL_NAME,
        "--host",                   "127.0.0.1",
        "--port",                   str(VLLM_PORT),
        "--max-model-len",          max_model_len,
        "--dtype",                  "auto",
        "--gpu-memory-utilization", gpu_mem_util,
        "--quantization",           "gptq_marlin",
        "--chat-template",          os.path.join(model_path, "chat_template.jinja"),
        "--reasoning-parser",       "qwen3",
    ]
    if os.environ.get("ENFORCE_EAGER", "0") == "1":
        cmd.append("--enforce-eager")
    cpu_offload = os.environ.get("CPU_OFFLOAD_GB", "0")
    if cpu_offload != "0":
        cmd += ["--cpu-offload-gb", cpu_offload]
    print(f"Starting vLLM with model: {model_path}", flush=True)
    return subprocess.Popen(cmd)


def wait_for_vllm():
    global vllm_ready
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2)
            vllm_ready = True
            print("vLLM ready", flush=True)
            return
        except Exception:
            time.sleep(1)
    print("ERROR: vLLM did not become ready within 600s", flush=True)


def vllm_request(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600).read()


class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200 if vllm_ready else 503)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not vllm_ready:
            self._json(503, {"error": "service unavailable"})
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.loads(body)
        except Exception as e:
            self._json(400, {"error": str(e)})
            return

        if self.path == "/invocations":
            self._handle_invocations(data)
        elif self.path in ("/v1/chat/completions", "/v1/completions"):
            # Transparent proxy: preserves stream, chat_template_kwargs, etc.
            self._proxy_vllm(self.path, body, data)
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_invocations(self, data):
        """Prompt-only endpoint per API contract: {"prompt":…, "max_tokens":…, "temperature":…}."""
        try:
            prompt = data.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            payload = json.dumps({
                "model":       SERVED_MODEL_NAME,
                "prompt":      prompt,
                "max_tokens":  data.get("max_tokens", 512),
                "temperature": data.get("temperature", 0.0),
            }).encode()
            result = vllm_request("/v1/completions", payload)
            resp   = json.loads(result)
            text   = resp["choices"][0]["text"]
            self._write(200, json.dumps({"choices": [{"text": text}]}).encode())
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _proxy_vllm(self, path, body, data):
        """Transparent proxy to vLLM — preserves streaming and all request fields."""
        is_stream = data.get("stream", False)

        # Always normalise model name to what vLLM is serving
        d = dict(data)
        d["model"] = SERVED_MODEL_NAME

        # Inject thinking mode if evaluator doesn't send chat_template_kwargs.
        # chat_template.jinja defaults to thinking ON; MMLU-Pro/IFEval need it OFF.
        # GPQA uses stream=True and needs thinking ON; others need it OFF.
        if path == "/v1/chat/completions" and "chat_template_kwargs" not in d:
            d["chat_template_kwargs"] = {"enable_thinking": bool(is_stream)}

        body = json.dumps(d).encode()

        conn = http.client.HTTPConnection("127.0.0.1", VLLM_PORT, timeout=600)
        try:
            conn.request("POST", path, body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()

            if is_stream:
                self.send_response(resp.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                result = resp.read()
                self._write(resp.status, result)
        except Exception as e:
            self._json(500, {"error": str(e)})
        finally:
            conn.close()

    def _json(self, code, obj):
        self._write(code, json.dumps(obj).encode())

    def _write(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    proc = start_vllm(MODEL_DIR)
    threading.Thread(target=wait_for_vllm, daemon=True).start()
    print(f"Listening on :{PORT} — waiting for vLLM...", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
