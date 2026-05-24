#!/usr/bin/env python3
"""
CPU-only inference server — mirrors GPU server's API contract exactly.
Endpoints: GET /ping, POST /invocations /v1/completions /v1/chat/completions
Thinking mode: controlled via chat_template_kwargs.enable_thinking (inferred from stream if absent).
"""
import json, re, threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import http.server

MODEL_DIR         = "/opt/ml/model"
PORT              = 8080
SERVED_MODEL_NAME = "Qwen/Qwen3.5-4B"

print("Loading model on CPU...", flush=True)
from gptqmodel import GPTQModel
from transformers import AutoTokenizer
import torch

tokenizer   = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model       = GPTQModel.load(MODEL_DIR, device="cpu")
model_ready = True
print(f"Model loaded. Listening on :{PORT}", flush=True)

gen_lock = threading.Lock()


def _prompt(messages, enable_thinking):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _generate(prompt, max_tokens, temperature=0.0):
    inputs    = tokenizer(prompt, return_tensors="pt")
    do_sample = temperature > 1e-6
    with gen_lock:
        with torch.no_grad():
            out = model.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200 if model_ready else 503)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not model_ready:
            self._json(503, {"error": "service unavailable"})
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.loads(body)
        except Exception as e:
            self._json(400, {"error": str(e)})
            return
        try:
            if self.path == "/invocations":
                self._invocations(data)
            elif self.path == "/v1/completions":
                self._completions(data)
            elif self.path == "/v1/chat/completions":
                self._chat(data)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── /invocations ─────────────────────────────────────────────────────────
    def _invocations(self, data):
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        text = _generate(prompt, data.get("max_tokens", 512), data.get("temperature", 0.0))
        self._json(200, {"choices": [{"text": text}]})

    # ── /v1/completions (latency benchmark) ──────────────────────────────────
    def _completions(self, data):
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        text = _generate(prompt, data.get("max_tokens", 512), data.get("temperature", 0.0))
        self._json(200, {
            "model":   SERVED_MODEL_NAME,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        })

    # ── /v1/chat/completions (quality benchmarks) ─────────────────────────────
    def _chat(self, data):
        messages    = data.get("messages", [])
        max_tokens  = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.0)
        is_stream   = data.get("stream", False)

        # thinking: from chat_template_kwargs if present, else infer from stream
        # (GPQA: stream=true → thinking ON; MMLU-Pro/IFEval: no stream → thinking OFF)
        ktw             = data.get("chat_template_kwargs", {})
        enable_thinking = ktw.get("enable_thinking", bool(is_stream))

        prompt  = _prompt(messages, enable_thinking)
        raw     = _generate(prompt, max_tokens, temperature)
        content = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

        if is_stream:
            self._sse(content)
        else:
            self._json(200, {
                "model":   SERVED_MODEL_NAME,
                "choices": [{
                    "index":         0,
                    "message":       {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
            })

    def _sse(self, content):
        """Emit SSE: one content chunk + stop chunk + [DONE]."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            chunk = json.dumps({
                "model":   SERVED_MODEL_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
            })
            self.wfile.write(f"data: {chunk}\n\n".encode())
            done = json.dumps({
                "model":   SERVED_MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            self.wfile.write(f"data: {done}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
