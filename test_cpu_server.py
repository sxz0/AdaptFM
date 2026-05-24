#!/usr/bin/env python3
"""
Smoke test for the CPU server against all three competition benchmark modes.
Usage: python3 test_cpu_server.py [base_url]
  base_url defaults to http://localhost:8080
"""
import http.client, json, sys, urllib.request, urllib.parse

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
host = urllib.parse.urlparse(BASE).hostname
port = urllib.parse.urlparse(BASE).port or 8080

passed = failed = 0

def ok(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}")
    if detail:
        print(f"         {detail}")

def post(path, data):
    req = urllib.request.Request(
        f"{BASE}{path}",
        json.dumps(data).encode(),
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600)

# ── 1. /ping ─────────────────────────────────────────────────────────────────
print("\n── /ping ──")
resp = urllib.request.urlopen(f"{BASE}/ping", timeout=10)
ok("/ping returns 200", resp.status == 200, f"status={resp.status}")

# ── 2. MMLU-Pro: chat, thinking OFF, no stream ────────────────────────────────
print("\n── MMLU-Pro (/v1/chat/completions, thinking OFF) ──")
resp = post("/v1/chat/completions", {
    "model":   "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Is Paris the capital of France? Reply with yes or no only."}],
    "max_tokens": 16,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
})
body = json.loads(resp.read())
content = body["choices"][0]["message"]["content"].strip().lower()
ok("response is single word yes/no", content in ("yes", "no"), f"content={repr(content)}")
ok("no think block in content", "<think>" not in content, f"content={repr(content[:80])}")

# ── 3. IFEval: chat, thinking OFF, no stream ─────────────────────────────────
print("\n── IFEval (/v1/chat/completions, thinking OFF) ──")
resp = post("/v1/chat/completions", {
    "model":   "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Write exactly 2 bullet points about Python. Start each with '- '."}],
    "max_tokens": 128,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
})
body = json.loads(resp.read())
content = body["choices"][0]["message"]["content"]
ok("response has bullet points", "- " in content, f"content={repr(content[:120])}")
ok("no think block in content", "<think>" not in content)
ok("finish_reason is stop", body["choices"][0].get("finish_reason") == "stop")

# ── 4. GPQA-Diamond: chat, thinking ON, stream=true ──────────────────────────
print("\n── GPQA-Diamond (/v1/chat/completions, thinking ON, stream=true) ──")
conn = http.client.HTTPConnection(host, port, timeout=600)
payload = json.dumps({
    "model":   "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Briefly explain Heisenberg's uncertainty principle."}],
    "max_tokens": 48,
    "temperature": 0.0,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": True},
}).encode()
conn.request("POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"})
r = conn.getresponse()
ok("SSE response 200", r.status == 200, f"status={r.status}")
ok("Content-Type is text/event-stream", "text/event-stream" in r.getheader("Content-Type", ""))
raw_sse = r.read().decode()
ok("contains 'data:' SSE lines", "data:" in raw_sse)
ok("ends with [DONE]", "[DONE]" in raw_sse)

# Collect content from SSE chunks
parts = []
for line in raw_sse.splitlines():
    if line.startswith("data:") and "[DONE]" not in line:
        try:
            chunk = json.loads(line[5:])
            parts.append(chunk["choices"][0]["delta"].get("content", ""))
        except Exception:
            pass
full_content = "".join(parts)
ok("SSE content is non-empty", len(full_content) > 0, f"content={repr(full_content[:100])}")
# With thinking ON and short max_tokens the think block may be truncated (no </think> yet) — that's fine
ok("no raw <think> tag in delivered content", "<think>" not in full_content)
conn.close()

# ── 5. Latency benchmark: /v1/completions (raw text) ─────────────────────────
print("\n── Latency (/v1/completions, raw text) ──")
resp = post("/v1/completions", {
    "model":       "Qwen/Qwen3.5-4B",
    "prompt":      "The capital of France is",
    "max_tokens":  16,
    "temperature": 0.0,
})
body = json.loads(resp.read())
text = body["choices"][0]["text"].strip()
ok("completions returns text", len(text) > 0, f"text={repr(text)}")

# ── 6. /invocations ──────────────────────────────────────────────────────────
print("\n── /invocations (legacy) ──")
resp = post("/invocations", {
    "prompt":      "The capital of Germany is",
    "max_tokens":  16,
    "temperature": 0.0,
})
body = json.loads(resp.read())
text = body["choices"][0]["text"].strip()
ok("invocations returns text", len(text) > 0, f"text={repr(text)}")

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Result: {passed}/{passed+failed} passed")
if failed:
    sys.exit(1)
