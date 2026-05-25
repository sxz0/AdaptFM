#!/usr/bin/env python3
"""
AdaptFM GPU inference server — vLLM + Qwen3.5 MTP speculative decoding.

Uses qwen3_5_mtp method (built-in MTP head in Qwen3.5-4B weights) for
native speculative decoding — predicts 1 extra token per forward pass,
effectively 1.5-2x decode speedup with high acceptance rate.

Also enables prefix caching for shared system prompts across requests.
"""
import json, os, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import http.server


class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


VLLM_PORT = 8081
MODEL_DIR  = "/opt/ml/model"
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-4B")
SERVED_MODEL_NAME = "default"
vllm_ready = False

# MTP speculative decoding config — uses the built-in MTP head in Qwen3.5-4B
# mtp_num_hidden_layers=1 → num_speculative_tokens=1 (one draft token per step)
# "mtp" is the non-deprecated replacement for "qwen3_5_mtp" in vLLM 0.19.0+
MTP_SPEC_CONFIG = json.dumps({
    "method": "mtp",
    "num_speculative_tokens": 1,
})


def resolve_model_path():
    if os.path.isdir(MODEL_DIR) and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        print(f"Model weights found at {MODEL_DIR}", flush=True)
        return MODEL_DIR
    return MODEL_NAME


def start_vllm(model_path):
    print(f"Starting vLLM with qwen3_5_mtp speculative decoding: {model_path}", flush=True)
    return subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--max-model-len", "16384",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",           # Free speedup for shared system prompts
        "--speculative-config", MTP_SPEC_CONFIG,   # qwen3_5_mtp native speculation
        "--chat-template", "/opt/ml/model/chat_template.jinja",
    ])


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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200 if vllm_ready else 503)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path in ("/invocations", "/v1/completions", "/v1/chat/completions"):
            try:
                data = json.loads(body)
                use_chat = "messages" in data and self.path != "/v1/completions"
                thinking = data.get("thinking", False)
                stream = data.get("stream", False)

                if use_chat and thinking:
                    payload = json.dumps({
                        "model": SERVED_MODEL_NAME,
                        "messages": data["messages"],
                        "max_tokens": data.get("max_tokens", 12288),
                        "temperature": data.get("temperature", 0.0),
                        "chat_template_kwargs": {"enable_thinking": True},
                        "stream": stream,
                    }).encode()
                    if stream:
                        self._proxy_stream("/v1/chat/completions", payload)
                        return
                    result = vllm_request("/v1/chat/completions", payload)
                elif use_chat:
                    payload = json.dumps({
                        "model": SERVED_MODEL_NAME,
                        "messages": data["messages"],
                        "max_tokens": data.get("max_tokens", 128),
                        "temperature": data.get("temperature", 0.0),
                        "chat_template_kwargs": {"enable_thinking": False},
                        "stop": data.get("stop", []),
                    }).encode()
                    result = vllm_request("/v1/chat/completions", payload)
                else:
                    prompt = data.get("prompt", "")
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else ""
                    payload = json.dumps({
                        "model": SERVED_MODEL_NAME,
                        "prompt": prompt,
                        "max_tokens": data.get("max_tokens", 128),
                        "temperature": data.get("temperature", 0.0),
                    }).encode()
                    result = vllm_request("/v1/completions", payload)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(result)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _proxy_stream(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        full_content = ""
        finish_reason = None
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        full_content += delta.get("content", "")
                        finish_reason = (chunk.get("choices", [{}])[0].get("finish_reason")
                                         or finish_reason)
                    except Exception:
                        pass
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        result = json.dumps({
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": finish_reason,
            }],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(result)


if __name__ == "__main__":
    model_path = resolve_model_path()
    proc = start_vllm(model_path)
    threading.Thread(target=wait_for_vllm, daemon=True).start()
    print("Listening on :8080 — waiting for vLLM + qwen3_5_mtp...", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
