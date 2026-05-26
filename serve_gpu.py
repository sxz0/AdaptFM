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

Architecture note:
  - Qwen3.5-4B uses GatedDeltaNet (hybrid recurrent) layers.
  - Speculative decoding (n-gram or draft model) is incompatible with recurrent
    architectures in ExLlamaV3 v0.0.37 — do NOT enable ngram_match_min.
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
# 131072 = 512 pages × 256 (PAGE_SIZE) — supports 8 concurrent 8K-token prompts +
# up to 12K output tokens per GPQA request without KV-page exhaustion.
MAX_CACHE_TOKENS  = int(os.environ.get("MAX_CACHE_TOKENS", "131072"))
NO_THINK_JINJA    = os.environ.get("NO_THINK_JINJA", "/opt/ml/qwen_no_think.jinja")

# ── Global state ──────────────────────────────────────────────────────────────
model_ready        = False
_generator         = None
_tokenizer         = None
_no_think_template = None
_stop_token_ids    = []   # populated after model load: [eos_id, im_end_id]

# Concurrency: HTTP handlers enqueue Jobs and block on their done_event.
# _gen_lock protects generator.enqueue() / generator.iterate() from racing.
_gen_lock     = threading.Lock()
_pending_jobs: dict = {}   # job_id -> (threading.Event, list)


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    global model_ready, _generator, _tokenizer, _no_think_template, _stop_token_ids

    from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator

    print(f"[serve] Loading model from: {MODEL_DIR}", flush=True)
    config     = Config.from_directory(MODEL_DIR)
    _tokenizer = Tokenizer.from_config(config)
    model      = Model.from_config(config)

    # Quantized KV cache (8-bit K, 8-bit V) — reduces KV memory bandwidth vs FP16
    # k_bits/v_bits passed as kwargs through Cache → CacheLayer_quant.__init__
    # k_bits/v_bits: 8-bit is the minimum safe value for Qwen3.5-4B.
    # 4-bit KV cache triggers AssertionError in recurrent_checkpoint() due to
    # GatedDeltaNet page-alignment requirements — do NOT go below 8.
    cache = Cache(model, max_num_tokens=MAX_CACHE_TOKENS,
                  layer_type=CacheLayer_quant, k_bits=8, v_bits=8)

    model.load(progressbar=True)

    # max_q_size controls how many jobs can be active simultaneously;
    # 16 is generous for our 8-concurrent eval.
    # NOTE: Qwen3.5-4B has GatedDeltaNet (hybrid recurrent) layers — speculative
    # decoding (ngram or draft model) is incompatible with recurrent architectures.
    _generator = Generator(model=model, cache=cache, tokenizer=_tokenizer,
                           max_q_size=16)

    # Determine stop token IDs (token IDs, not strings, to avoid scanning overhead).
    # Qwen3.5-4B: eos = 248044 (<|endoftext|>), im_end = 248046 (<|im_end|>)
    im_end_enc = _tokenizer.encode("<|im_end|>", add_bos=False)
    im_end_id  = int(im_end_enc[0, -1]) if im_end_enc.numel() > 0 else None
    _stop_token_ids = list({_tokenizer.eos_token_id, im_end_id} - {None})
    print(f"[serve] Stop tokens: {_stop_token_ids}", flush=True)

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
    """
    Strip thinking content from the model output.

    Qwen3.5 thinking mode: the chat template appends '<think>\\n' to the prompt,
    so the model's generated text starts with the REASONING CONTENT (no opening
    <think> tag) and ends with '</think>\\n\\nanswer'.  We need to handle this case
    as well as the fallback where <think>...</think> appears inline.
    """
    # Case 1: inline <think>…</think> block (e.g. full_completion has both tags)
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if stripped != text:
        return stripped.strip()
    # Case 2: prompt ended with '<think>\n', output starts with reasoning until </think>
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()
    # Case 3: no </think> found — return as-is (model may not have started thinking)
    return text.strip()


# ── Generator thread ──────────────────────────────────────────────────────────
def _generator_thread():
    """
    Single thread that drives the ExLlamaV3 generator.
    - Calls iterate() under _gen_lock so HTTP threads can safely enqueue between calls.
    - When a job finishes (eos=True), wakes the waiting HTTP handler via its Event.
    - Idles at 1 ms poll when the queue is empty (overhead ≪ generation latency).
    - On unexpected exception: wakes all pending jobs with an error, then continues.
    """
    # Wait until the model finishes loading before entering the main loop
    while not model_ready:
        time.sleep(0.05)

    while True:
        try:
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

        except Exception as exc:
            # Generator error: unblock all pending HTTP handlers with the error,
            # then continue — this keeps the server alive for subsequent requests.
            print(f"[serve] Generator error: {exc}", flush=True)
            import traceback; traceback.print_exc()
            for job_id in list(_pending_jobs.keys()):
                entry = _pending_jobs.pop(job_id, None)
                if entry:
                    done_event, holder = entry
                    if not holder:
                        holder.append({"text": "", "error": str(exc)})
                    done_event.set()
            time.sleep(0.1)   # brief back-off before retrying


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
        stop_conditions=_stop_token_ids,   # token IDs only — faster than string matching
        identifier=job_id,
    )

    done_event = threading.Event()
    holder: list = []
    _pending_jobs[job_id] = (done_event, holder)

    with _gen_lock:
        _generator.enqueue(job)

    done_event.wait(timeout=600)   # 10-minute safety timeout

    if not holder:
        # Timed out or generator thread crashed without waking us
        _pending_jobs.pop(job_id, None)
        raise RuntimeError("generation timed out or generator crashed")

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
                # Chat-mode invocation: apply template + optionally strip thinking.
                # Return both 'text' (competition spec) and 'message.content'
                # (eval-script compat) so both clients work.
                thinking = data.get("thinking", False)
                ctk = data.get("chat_template_kwargs", {})
                if ctk.get("enable_thinking") is False:
                    thinking = False
                prompt_text  = _apply_chat_template(data["messages"], enable_thinking=thinking)
                raw_text     = _infer(prompt_text, max_tokens, temperature)
                visible_text = _strip_think(raw_text) if thinking else raw_text.strip()
                self._json(200, {"choices": [{
                    "text": visible_text,
                    "message": {"role": "assistant", "content": visible_text},
                }]})
            else:
                # Raw-prompt invocation (latency benchmark uses this form)
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
