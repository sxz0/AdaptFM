#!/usr/bin/env python3
"""
AdaptFM GPU inference server — llama.cpp / GGUF edition.

Loads a GGUF-quantised Qwen3.5-4B from /opt/ml/model/model.gguf and serves
the competition API on port 8080.

  GET  /ping                 → 200 when ready, 503 while loading
  POST /invocations          → {"prompt":…, "max_tokens":…, "temperature":…}
                                or {"messages":[…], …}
                                Returns {"choices":[{"text":…}]}
  POST /v1/completions       → OpenAI raw completions
  POST /v1/chat/completions  → OpenAI chat completions (thinking mode support)
"""

import http.server
import json
import os
import queue
import re
import threading
import time
import uuid
from socketserver import ThreadingMixIn

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH        = os.environ.get("MODEL_PATH", "/opt/ml/model/model.gguf")
PORT              = int(os.environ.get("PORT", "8080"))
SERVED_MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-4B")
N_CTX             = int(os.environ.get("N_CTX", "32768"))
N_GPU_LAYERS      = int(os.environ.get("N_GPU_LAYERS", "-1"))   # -1 = all on GPU

# ChatML tokens
BOS = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"

# ── Global state ───────────────────────────────────────────────────────────────
model_ready = False
_llm        = None
_req_queue: queue.Queue = queue.Queue()


# ── Prompt formatting ──────────────────────────────────────────────────────────
def _format_prompt(messages: list, enable_thinking: bool = False) -> str:
    """
    Build a ChatML-formatted prompt for Qwen3.5.

    With enable_thinking=False the assistant turn is pre-seeded with an empty
    <think> block so the model skips chain-of-thought.
    """
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{BOS}{role}\n{content}{EOS_TOKEN}\n")
    parts.append(f"{BOS}assistant\n")

    if not enable_thinking:
        parts.append("<think>\n\n</think>\n\n")

    return "".join(parts)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model():
    global model_ready, _llm
    from llama_cpp import Llama

    print(f"[serve] Loading GGUF from: {MODEL_PATH}", flush=True)
    _llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,   # -1 → offload all layers to A10G
        n_ctx=N_CTX,
        n_batch=512,
        flash_attn=True,             # use Flash Attention in llama.cpp
        verbose=False,
    )
    model_ready = True
    print("[serve] Model ready — listening on port 8080", flush=True)


# ── Generator thread ───────────────────────────────────────────────────────────
def _generator_thread():
    """Serialise inference through a single thread to avoid CUDA races."""
    while True:
        item = _req_queue.get()
        if item is None:
            break

        req, done_event, result_holder = item
        try:
            output = _llm.create_completion(
                prompt=req["prompt"],
                max_tokens=req.get("max_tokens", 512),
                temperature=req.get("temperature", 0.0),
                top_p=0.9 if req.get("temperature", 0.0) > 0 else 1.0,
                stop=[EOS_TOKEN],
                echo=False,
            )
            text = output["choices"][0]["text"]
            result_holder.append({"text": text, "error": None})
        except Exception as e:
            result_holder.append({"text": "", "error": str(e)})
        finally:
            done_event.set()


def _infer(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    done  = threading.Event()
    holder = []
    _req_queue.put((
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        done, holder,
    ))
    done.wait()
    r = holder[0]
    if r["error"]:
        raise RuntimeError(r["error"])
    return r["text"]


# ── HTTP server ────────────────────────────────────────────────────────────────
class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200 if model_ready else 503)
            self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if not model_ready:
            return self._json(503, {"error": "model loading"})
        try:
            n    = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
        except Exception as e:
            return self._json(400, {"error": str(e)})

        if self.path == "/invocations":
            self._invocations(data)
        elif self.path == "/v1/completions":
            self._completions(data)
        elif self.path == "/v1/chat/completions":
            self._chat(data)
        else:
            self.send_response(404); self.end_headers()

    def _invocations(self, data):
        try:
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))
            if "messages" in data:
                thinking = data.get("thinking", False)
                if data.get("chat_template_kwargs", {}).get("enable_thinking") is False:
                    thinking = False
                prompt = _format_prompt(data["messages"], enable_thinking=thinking)
            else:
                prompt = data.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""
            text = _infer(prompt, max_tokens, temperature)
            self._json(200, {"choices": [{"text": text}]})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _completions(self, data):
        try:
            prompt = data.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            text = _infer(prompt, int(data.get("max_tokens", 512)),
                          float(data.get("temperature", 0.0)))
            self._json(200, {
                "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
                "created": int(time.time()), "model": SERVED_MODEL_NAME,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _chat(self, data):
        try:
            messages    = data.get("messages", [])
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))
            is_stream   = data.get("stream", False)
            thinking    = data.get("thinking", False)
            if data.get("chat_template_kwargs", {}).get("enable_thinking") is False:
                thinking = False
            elif is_stream:
                thinking = True   # GPQA eval uses stream=True + wants reasoning

            prompt       = _format_prompt(messages, enable_thinking=thinking)
            text         = _infer(prompt, max_tokens, temperature)
            visible_text = _strip_think(text) if thinking else text.strip()

            req_id = f"chatcmpl-{uuid.uuid4().hex}"
            ts     = int(time.time())

            if is_stream:
                chunks = [
                    {"id": req_id, "object": "chat.completion.chunk",
                     "created": ts, "model": SERVED_MODEL_NAME,
                     "choices": [{"index": 0,
                                  "delta": {"role": "assistant", "content": visible_text},
                                  "finish_reason": None}]},
                    {"id": req_id, "object": "chat.completion.chunk",
                     "created": ts, "model": SERVED_MODEL_NAME,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._json(200, {
                    "id": req_id, "object": "chat.completion",
                    "created": ts, "model": SERVED_MODEL_NAME,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": visible_text},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_generator_thread, daemon=True).start()
    threading.Thread(target=_load_model, daemon=True).start()
    print(f"[serve] HTTP server listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
