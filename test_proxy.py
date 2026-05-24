#!/usr/bin/env python3
"""
Smoke test: mock vLLM on 8081, run serve_gpu.py proxy, verify ping and invocations.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

MOCK_PORT = 8081
PROXY_PORT = 8082  # avoid conflict with any running service


class MockVLLM(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        data = json.loads(body)
        if "messages" in data:
            resp = json.dumps({"choices": [{"message": {"role": "assistant", "content": "test response"}}]})
        else:
            resp = json.dumps({"choices": [{"text": "test response"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp.encode())


def start_mock():
    srv = HTTPServer(("127.0.0.1", MOCK_PORT), MockVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Mock vLLM running on :{MOCK_PORT}", flush=True)


def run_tests():
    base = f"http://127.0.0.1:{PROXY_PORT}"

    # ping before ready → 503
    resp = urllib.request.urlopen(f"{base}/ping")
    assert resp.status == 200, f"Expected 200, got {resp.status}"
    print("PASS: /ping returns 200 (proxy running, vLLM mock ready)", flush=True)

    # chat completions
    payload = json.dumps({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }).encode()
    req = urllib.request.Request(
        f"{base}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    body = json.loads(resp.read())
    assert body["choices"][0]["message"]["content"] == "test response"
    print("PASS: POST /invocations (chat) returns correct response", flush=True)

    # completions
    payload = json.dumps({"prompt": "hello", "max_tokens": 10}).encode()
    req = urllib.request.Request(
        f"{base}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    body = json.loads(resp.read())
    assert body["choices"][0]["text"] == "test response"
    print("PASS: POST /invocations (completion) returns correct response", flush=True)


if __name__ == "__main__":
    start_mock()

    # Patch serve_gpu.py to use our ports and skip vLLM subprocess
    import importlib.util, types, os
    os.environ["_TEST_PROXY_ONLY"] = "1"

    # Run a minimal version of the proxy pointing at mock vLLM
    import http.server
    from socketserver import ThreadingMixIn

    sys.path.insert(0, os.path.dirname(__file__))

    # Monkey-patch constants in serve_gpu before import
    import serve_gpu
    serve_gpu.VLLM_PORT = MOCK_PORT
    serve_gpu.PORT = PROXY_PORT
    serve_gpu.vllm_ready = True  # pretend vLLM is already ready

    srv = serve_gpu.ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), serve_gpu.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Proxy running on :{PROXY_PORT}", flush=True)

    time.sleep(0.3)
    try:
        run_tests()
        print("\nAll tests PASSED", flush=True)
    except Exception as e:
        print(f"\nFAIL: {e}", flush=True)
        sys.exit(1)
    srv.shutdown()
