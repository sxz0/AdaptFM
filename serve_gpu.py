#!/usr/bin/env python3
"""
AdaptFM GPU inference server — ExLlamaV3 edition.

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

Concurrency: requests are serialised through a single generator thread to avoid
CUDA race conditions; multiple HTTP threads enqueue jobs and block until done.
"""

import http.server
import json
import os
import queue
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
NO_THINK_JINJA    = os.environ.get(
    "NO_THINK_JINJA", "/opt/ml/qwen_no_think.jinja"
)

# ── Global state ──────────────────────────────────────────────────────────────
model_ready  = False
_generator   = None
_tokenizer   = None
_hf_tok      = None      # underlying HF tokenizer for apply_chat_template
_no_think_template = None
_req_queue: queue.Queue = queue.Queue()   # (request_dict, Event, result_holder)


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    global model_ready, _generator, _tokenizer, _hf_tok, _no_think_template

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator

    print(f"[serve] Loading model from: {MODEL_DIR}", flush=True)
    config    = Config.from_directory(MODEL_DIR)
    _tokenizer = Tokenizer.from_config(config)
    _hf_tok    = _tokenizer.tokenizer   # HF tokenizer with apply_chat_template
    model     = Model.from_config(config)
    cache     = Cache(model, max_num_tokens=MAX_CACHE_TOKENS)
    model.load(progressbar=True)
    _generator = Generator(model=model, cache=cache, tokenizer=_tokenizer)

    # Load no-think jinja template if available
    if os.path.isfile(NO_THINK_JINJA):
        with open(NO_THINK_JINJA) as f:
            _no_think_template = f.read()
        print(f"[serve] Loaded no-think template from {NO_THINK_JINJA}", flush=True)

    model_ready = True
    print("[serve] Model ready — listening on port 8080", flush=True)


# ── Chat template helpers ─────────────────────────────────────────────────────
def _apply_chat_template(messages: list, enable_thinking: bool = False) -> str:
    """
    Apply Qwen3.5 chat template.

    enable_thinking=True  → use the default template (thinking ON)
    enable_thinking=False → force empty <think></think> block (thinking OFF)
    """
    if enable_thinking:
        prompt = _hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Use no-think jinja if available; otherwise fall back to empty think block
        if _no_think_template:
            prompt = _hf_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=_no_think_template,
            )
        else:
            prompt = _hf_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Append empty think block to suppress thinking
            prompt += "<think>\n\n</think>\n\n"
    return prompt


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ── Generator thread ──────────────────────────────────────────────────────────
def _generator_thread():
    """
    Single thread that owns the ExLlamaV3 generator.
    Reads (request_dict, done_event, result_list) tuples from _req_queue.
    """
    while True:
        item = _req_queue.get()
        if item is None:
            break  # shutdown

        req, done_event, result_holder = item
        try:
            from exllamav3 import TopPSampler

            prompt_text = req["prompt_text"]
            max_new_tokens = req.get("max_new_tokens", 512)
            temperature    = float(req.get("temperature", 0.0))

            sampler = TopPSampler(
                top_p=0.9 if temperature > 0 else 1.0,
                temperature=temperature,
            )

            # Stop on <|im_end|> and EOS
            stop_conditions = [_tokenizer.eos_token_id, "<|im_end|>"]

            text = _generator.generate(
                prompt=prompt_text,
                max_new_tokens=max_new_tokens,
                sampler=sampler,
                stop_conditions=stop_conditions,
                add_bos=True,
                completion_only=True,
            )
            result_holder.append({"text": text, "error": None})
        except Exception as e:
            result_holder.append({"text": "", "error": str(e)})
        finally:
            done_event.set()


def _infer(prompt_text: str, max_new_tokens: int = 512, temperature: float = 0.0):
    """
    Thread-safe inference: enqueue the request and block until done.
    """
    done_event    = threading.Event()
    result_holder = []
    _req_queue.put(({
        "prompt_text":   prompt_text,
        "max_new_tokens": max_new_tokens,
        "temperature":    temperature,
    }, done_event, result_holder))
    done_event.wait()
    r = result_holder[0]
    if r["error"]:
        raise RuntimeError(r["error"])
    return r["text"]


# ── HTTP server ───────────────────────────────────────────────────────────────
class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/ping":
            code = 200 if model_ready else 503
            self.send_response(code)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        if not model_ready:
            self._json(503, {"error": "service unavailable — model loading"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body)
        except Exception as e:
            self._json(400, {"error": f"bad request: {e}"})
            return

        path = self.path
        if path == "/invocations":
            self._handle_invocations(data)
        elif path == "/v1/completions":
            self._handle_raw_completions(data)
        elif path == "/v1/chat/completions":
            self._handle_chat_completions(data)
        else:
            self.send_response(404)
            self.end_headers()

    # ── /invocations ─────────────────────────────────────────────────────────
    def _handle_invocations(self, data):
        """
        Competition invocations endpoint.
        Input:  {"prompt": str, "max_tokens": int, "temperature": float}
        Also accepts: {"messages": [...], "max_tokens": int, ...} (chat format)
        Output: {"choices": [{"text": str}]}
        """
        try:
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))

            if "messages" in data:
                # Chat invocation — evaluate thinking flag
                thinking = data.get("thinking", False)
                ctk = data.get("chat_template_kwargs", {})
                if ctk.get("enable_thinking") is False:
                    thinking = False
                prompt_text = _apply_chat_template(data["messages"], enable_thinking=thinking)
            else:
                # Raw completion
                prompt_text = data.get("prompt", "")
                if isinstance(prompt_text, list):
                    prompt_text = prompt_text[0] if prompt_text else ""

            text = _infer(prompt_text, max_tokens, temperature)
            self._json(200, {"choices": [{"text": text}]})
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── /v1/completions ───────────────────────────────────────────────────────
    def _handle_raw_completions(self, data):
        """OpenAI-compatible raw completions (no chat template)."""
        try:
            prompt      = data.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))

            text = _infer(prompt, max_tokens, temperature)

            resp = {
                "id":      f"cmpl-{uuid.uuid4().hex}",
                "object":  "text_completion",
                "created": int(time.time()),
                "model":   SERVED_MODEL_NAME,
                "choices": [{
                    "index":         0,
                    "text":          text,
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            self._json(200, resp)
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── /v1/chat/completions ──────────────────────────────────────────────────
    def _handle_chat_completions(self, data):
        """OpenAI-compatible chat completions."""
        try:
            messages    = data.get("messages", [])
            max_tokens  = int(data.get("max_tokens", 512))
            temperature = float(data.get("temperature", 0.0))
            is_stream   = data.get("stream", False)

            # Determine thinking mode
            thinking = data.get("thinking", False)
            ctk = data.get("chat_template_kwargs", {})
            if ctk.get("enable_thinking") is False:
                thinking = False
            elif is_stream:
                # GPQA-style eval uses stream=True and wants thinking
                thinking = True

            prompt_text = _apply_chat_template(messages, enable_thinking=thinking)
            text        = _infer(prompt_text, max_tokens, temperature)

            # Remove think blocks from output when thinking was enabled
            if thinking:
                visible_text = _strip_think(text)
            else:
                visible_text = text.strip()

            req_id = f"chatcmpl-{uuid.uuid4().hex}"
            ts     = int(time.time())

            if is_stream:
                # SSE stream — one chunk then done
                chunks = [
                    {"id": req_id, "object": "chat.completion.chunk", "created": ts,
                     "model": SERVED_MODEL_NAME,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": visible_text},
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
                    line = f"data: {json.dumps(chunk)}\n\n".encode()
                    self.wfile.write(line)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                resp = {
                    "id":      req_id,
                    "object":  "chat.completion",
                    "created": ts,
                    "model":   SERVED_MODEL_NAME,
                    "choices": [{
                        "index":   0,
                        "message": {"role": "assistant", "content": visible_text},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                self._json(200, resp)
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── helpers ───────────────────────────────────────────────────────────────
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start generator thread (owns CUDA context)
    gen_thread = threading.Thread(target=_generator_thread, daemon=True)
    gen_thread.start()

    # Load model in a background thread so we can start the HTTP server immediately
    loader = threading.Thread(target=_load_model, daemon=True)
    loader.start()

    print(f"[serve] HTTP server listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
