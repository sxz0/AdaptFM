#!/usr/bin/env python3
"""
AdaptFM GPU inference server — ExLlamaV3 edition (v2: concurrent + Q-KV cache).

Loads an EXL3-quantized (or plain FP16/HF) Qwen3.5-4B from /opt/ml/model and
serves the competition API on port 8080:

  GET  /ping              → 200 when model ready, 503 while loading
  POST /invocations       → {"prompt":…,"max_tokens":…,"temperature":…}
                             Returns {"choices":[{"text":…}]}
  POST /v1/completions    → OpenAI-compatible raw completions
  POST /v1/chat/completions → OpenAI-compatible chat with thinking mode support

Thinking control (chat endpoint):
  - payload has "thinking": true  → enable Qwen3.5 chain-of-thought
  - payload has "chat_template_kwargs": {"enable_thinking": false}  → disable thinking
  - stream=True defaults to thinking ON (for GPQA-style eval)
  - stream=False defaults to thinking OFF

Concurrency v2:
  - Uses Generator.enqueue(Job) + iterate() instead of generate()
  - Multiple HTTP threads can enqueue jobs simultaneously
  - Generator processes all active jobs concurrently each iterate() call
  - 1 ms idle poll (negligible latency overhead vs 1–4 s generation time)

KV Cache:
  - Uses CacheLayer_quant (quantized KV cache) to reduce memory bandwidth
  - Especially beneficial for medium/long contexts
"""

import http.server
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR         = os.environ.get("MODEL_DIR", "/opt/ml/model")
PORT              = int(os.environ.get("PORT", "8080"))
SERVED_MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-4B")
MAX_CACHE_TOKENS  = int(os.environ.get("MAX_CACHE_TOKENS", "32768"))
NO_THINK_JINJA    = os.environ.get("NO_THINK_JINJA", "/opt/ml/qwen_no_think.jinja")

# ── Global state ──────────────────────────────────────────────────────────────
model_ready        = False
_generator         = None
_tokenizer         = None
_no_think_template = None

# Concurrency: HTTP handlers enqueue Jobs and block on their done_event.
# _gen_lock protects generator.enqueue() / generator.iterate() from racing.
_gen_lock     = threading.Lock()
_pending_jobs: dict = {}   # job_id -> (threading.Event, list)


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    global model_ready, _generator, _tokenizer, _no_think_template

    from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator

    print(f"[serve] Loading model from: {MODEL_DIR}", flush=True)
    config     = Config.from_directory(MODEL_DIR)
    _tokenizer = Tokenizer.from_config(config)
    model      = Model.from_config(config)

    # Quantized KV cache (8-bit K, 8-bit V) — reduces KV memory bandwidth vs FP16
    # k_bits/v_bits passed as kwargs through Cache → CacheLayer_quant.__init__
    cache = Cache(model, max_num_tokens=MAX_CACHE_TOKENS,
                  layer_type=CacheLayer_quant, k_bits=8, v_bits=8)

    model.load(progressbar=True)

    # max_q_size controls how many jobs can be active simultaneously;
    # 16 is generous for our 8-concurrent eval.
    _generator = Generator(model=model, cache=cache, tokenizer=_tokenizer,
                           max_q_size=16)

    if os.path.isfile(NO_THINK_JINJA):
        with open(NO_THINK_JINJA) as f:
            _no_think_template = f.read()
        print(f"[serve] Loaded no-think template from {NO_THINK_JINJA}", flush=True)

    model_ready = True
    print("[serve] Model ready — listening on port 8080", flush=True)


# ── Chat template helpers ─────────────────────────────────────────────────────
def _apply_chat_template(messages: list, enable_thinking: bool = False) -> str:
    if enable_thinking:
        return _tokenizer.hf_render_chat_template(messages, add_generation_prompt=True)
    if _no_think_template:
        return _tokenizer.hf_render_chat_template(
            messages, add_generation_prompt=True, chat_template=_no_think_template)
    prompt = _tokenizer.hf_render_chat_template(messages, add_generation_prompt=True)
    return prompt + "<think>\n\n</think>\n\n"


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ── Generator thread ──────────────────────────────────────────────────────────
def _generator_thread():
    """
    Single thread that drives the ExLlamaV3 generator.
    - Calls iterate() under _gen_lock so HTTP threads can safely enqueue between calls.
    - When a job finishes (eos=True), wakes the waiting HTTP handler via its Event.
    - Idles at 1 ms poll when the queue is empty (overhead ≪ generation latency).
    """
    # Wait until the model finishes loading before entering the main loop
    while not model_ready:
        time.sleep(0.05)

    while True:
        with _gen_lock:
            remaining = _generator.num_remaining_jobs()
            if remaining > 0:
                results = _generator.iterate()
            else:
                results = []

        for result in results:
            if result.get("stage") == "streaming" and result.get("eos"):
                job     = result["job"]
                job_id  = job.identifier
                text    = result.get("full_completion", "")
                entry   = _pending_jobs.pop(job_id, None)
                if entry:
                    done_event, holder = entry
                    holder.append({"text": text, "error": None})
                    done_event.set()

        if not results:
            time.sleep(0.001)   # 1 ms idle poll


# ── Thread-safe inference ─────────────────────────────────────────────────────
def _infer(prompt_text: str, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Enqueue a generation job and block until it completes.
    Multiple callers run concurrently — the generator processes all active jobs
    in parallel during each iterate() call.
    """
    import torch
    from exllamav3 import Job, TopPSampler

    # Tokenise (encode returns shape (1, seq_len))
    input_ids = _tokenizer.encode(prompt_text, add_bos=True)

    sampler = TopPSampler(
        top_p=0.9 if temperature > 0 else 1.0,
        temperature=temperature,
    )

    job_id = str(uuid.uuid4())
    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        sampler=sampler,
        stop_conditions=[_tokenizer.eos_token_id, "<|im_end|>"],
        identifier=job_id,
    )

    done_event = threading.Event()
    holder: list = []
    _pending_jobs[job_id] = (done_event, holder)

    with _gen_lock:
        _generator.enqueue(job)

    done_event.wait()

    r = holder[0]
    if r["error"]:
        raise RuntimeError(r["error"])
    return r["text"]


# ── HTTP server ───────────────────────────────────────────────────────────────
class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
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
            self._json(503, {"error": "service unavailable — model loading"}); return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.loads(body)
        except Exception as e:
            self._json(400, {"error": f"bad request: {e}"}); return

        if   self.path == "/invocations":        self._handle_invocations(data)
        elif self.path == "/v1/completions":      self._handle_raw_completions(data)
        elif self.path == "/v1/chat/completions": self._handle_chat_completions(data)
        else:
            self.send_response(404); self.end_headers()

    # ── /invocations ──────────────────────────────────────────────────────────
    def _handle_invocations(self, data):
        try:
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))

            if "messages" in data:
                thinking = data.get("thinking", False)
                ctk = data.get("chat_template_kwargs", {})
                if ctk.get("enable_thinking") is False:
                    thinking = False
                prompt_text = _apply_chat_template(data["messages"], enable_thinking=thinking)
            else:
                prompt_text = data.get("prompt", "")
                if isinstance(prompt_text, list):
                    prompt_text = prompt_text[0] if prompt_text else ""

            text = _infer(prompt_text, max_tokens, temperature)
            self._json(200, {"choices": [{"text": text}]})
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── /v1/completions ───────────────────────────────────────────────────────
    def _handle_raw_completions(self, data):
        try:
            prompt = data.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))
            text = _infer(prompt, max_tokens, temperature)
            self._json(200, {
                "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
                "created": int(time.time()), "model": SERVED_MODEL_NAME,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── /v1/chat/completions ──────────────────────────────────────────────────
    def _handle_chat_completions(self, data):
        try:
            messages    = data.get("messages", [])
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))
            is_stream   = data.get("stream", False)

            thinking = data.get("thinking", False)
            ctk = data.get("chat_template_kwargs", {})
            if ctk.get("enable_thinking") is False:
                thinking = False
            elif is_stream:
                thinking = True

            prompt_text  = _apply_chat_template(messages, enable_thinking=thinking)
            text         = _infer(prompt_text, max_tokens, temperature)
            visible_text = _strip_think(text) if thinking else text.strip()

            req_id = f"chatcmpl-{uuid.uuid4().hex}"
            ts     = int(time.time())

            if is_stream:
                chunks = [
                    {"id": req_id, "object": "chat.completion.chunk", "created": ts,
                     "model": SERVED_MODEL_NAME,
                     "choices": [{"index": 0, "delta": {"role": "assistant",
                                                         "content": visible_text},
                                  "finish_reason": None}]},
                    {"id": req_id, "object": "chat.completion.chunk", "created": ts,
                     "model": SERVED_MODEL_NAME,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start generator thread (drives ExLlamaV3 iterate loop)
    threading.Thread(target=_generator_thread, daemon=True).start()

    # Load model in background so HTTP server starts immediately
    threading.Thread(target=_load_model, daemon=True).start()

    print(f"[serve] HTTP server listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
