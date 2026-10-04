#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import math
import os
import random
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

VERSION = "1.0.0"
AUTHOR = "Aissa Mohammedi (DGK)"
LICENSE_NAME = "NEXUS-OPEN-2.0"

HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "nexus-cuda-killer")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, "logs", "nexus_cuda_killer.log")
PORT = 8103

def log(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def softmax_naive(values):
    exps = [math.exp(v) for v in values]
    total = sum(exps)
    return [x / total for x in exps]

def softmax_online(values):
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [x / total for x in exps]

def benchmark_softmax(n=32000, iterations=5):
    values = [random.uniform(-5, 5) for _ in range(n)]
    start = time.perf_counter()
    for _ in range(iterations):
        softmax_naive(values)
    naive_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        softmax_online(values)
    online_time = time.perf_counter() - start

    return {
        "ok": True,
        "n": n,
        "iterations": iterations,
        "naive_time_s": round(naive_time, 6),
        "online_time_s": round(online_time, 6),
        "speedup": round(naive_time / online_time, 4) if online_time > 0 else 0.0,
        "summary": "Online softmax reduces redundant memory passes and keeps a running max/sum."
    }

def sha256_hex(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def hash_batch(messages):
    return [sha256_hex(m) for m in messages]

def merkle_root(leaves):
    if not leaves:
        return ""
    level = [l.strip() for l in leaves]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level = []
        for i in range(0, len(level), 2):
            next_level.append(sha256_hex(level[i] + level[i + 1]))
        level = next_level
    return level[0]

def hash_summary(messages):
    hashes = hash_batch(messages)
    root = merkle_root(hashes)
    return {
        "ok": True,
        "count": len(messages),
        "hashes": hashes,
        "merkle_root": root
    }

def build_merkle_tree(leaves):
    if not leaves:
        return {"ok": False, "error": "No leaves provided"}
    levels = [leaves[:]]
    current = leaves[:]
    while len(current) > 1:
        if len(current) % 2 == 1:
            current.append(current[-1])
        nxt = []
        for i in range(0, len(current), 2):
            nxt.append(sha256_hex(current[i] + current[i + 1]))
        levels.append(nxt)
        current = nxt
    return {"ok": True, "levels": levels, "root": current[0]}

def generate_matrix(n, seed=0):
    random.seed(seed)
    return [[random.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]

def matmul(A, B):
    n = len(A)
    m = len(B[0])
    p = len(B)
    out = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            out[i][j] = s
    return out

def simulate_llm_inference(params_b=7.0, tokens=1000, batch=1):
    flops_per_token = 2 * params_b * 1e9
    effective_tflops = 800.0
    seconds_per_token = flops_per_token / (effective_tflops * 1e12)
    total_seconds = seconds_per_token * tokens * batch
    return {
        "ok": True,
        "params_b": params_b,
        "tokens": tokens,
        "batch": batch,
        "effective_tflops": effective_tflops,
        "seconds_total": round(total_seconds, 6),
        "seconds_per_token": round(seconds_per_token, 6),
        "tokens_per_second": round((tokens * batch) / total_seconds, 2) if total_seconds > 0 else 0.0
    }

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NEXUS CUDA KILLER</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 24px; }
    .container { max-width: 1000px; margin: 0 auto; }
    h1 { color: #7dd3fc; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 18px; margin-bottom: 16px; }
    pre { background: #0b1220; border: 1px solid #30363d; border-radius: 8px; padding: 12px; overflow: auto; }
    button { background: #238636; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; }
  </style>
</head>
<body>
  <div class="container">
    <h1>NEXUS CUDA KILLER</h1>
    <div class="card">
      <p>Portable Python simulator of GPU optimization logic.</p>
      <button onclick="loadBenchmark()">Run softmax benchmark</button>
      <button onclick="loadHash()">Run hash benchmark</button>
      <button onclick="loadMatrix()">Run matrix simulation</button>
    </div>

    <div class="card">
      <h2>Softmax Benchmark</h2>
      <pre id="softmax-output">Waiting...</pre>
    </div>

    <div class="card">
      <h2>Hash Benchmark</h2>
      <pre id="hash-output">Waiting...</pre>
    </div>

    <div class="card">
      <h2>Matrix Simulation</h2>
      <pre id="matrix-output">Waiting...</pre>
    </div>
  </div>

  <script>
    async function loadBenchmark() {
      const res = await fetch('/api/softmax-benchmark');
      const data = await res.json();
      document.getElementById('softmax-output').textContent = JSON.stringify(data, null, 2);
    }
    async function loadHash() {
      const res = await fetch('/api/hash-benchmark');
      const data = await res.json();
      document.getElementById('hash-output').textContent = JSON.stringify(data, null, 2);
    }
    async function loadMatrix() {
      const res = await fetch('/api/matrix-sim');
      const data = await res.json();
      document.getElementById('matrix-output').textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ['/', '/index.html']:
            data = HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == '/api/softmax-benchmark':
            self._send_json(benchmark_softmax())
            return

        if self.path == '/api/hash-benchmark':
            msgs = [f'nexus-{i}' for i in range(16)]
            self._send_json(hash_summary(msgs))
            return

        if self.path == '/api/matrix-sim':
            A = generate_matrix(8, seed=42)
            B = generate_matrix(8, seed=7)
            out = matmul(A, B)
            self._send_json({
                'ok': True,
                'shape': [len(out), len(out[0])],
                'sample': out[0][:3],
                'summary': 'Simulated matrix multiply for GPU-style compute workload.'
            })
            return

        if self.path == '/api/health':
            self._send_json({'ok': True, 'version': VERSION, 'author': AUTHOR})
            return

        self._send_json({'ok': False, 'error': 'Not found'}, 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length).decode('utf-8')
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        if self.path == '/api/softmax':
            values = payload.get('values', [0.2, 0.8, 1.3, 2.1])
            self._send_json({'ok': True, 'naive': softmax_naive(values), 'online': softmax_online(values)})
            return

        if self.path == '/api/hash':
            messages = payload.get('messages', ['nexus', 'cuda', 'killer'])
            self._send_json(hash_summary(messages))
            return

        if self.path == '/api/merkle':
            leaves = payload.get('leaves', ['a', 'b', 'c', 'd'])
            self._send_json(build_merkle_tree(leaves))
            return

        if self.path == '/api/llm':
            params_b = float(payload.get('params_b', 7.0))
            tokens = int(payload.get('tokens', 1000))
            batch = int(payload.get('batch', 1))
            self._send_json(simulate_llm_inference(params_b, tokens, batch))
            return

        self._send_json({'ok': False, 'error': 'Unknown endpoint'}, 404)


def find_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return 'localhost'


def main():
    print('=' * 72)
    print(f'NEXUS CUDA KILLER v{VERSION}')
    print(f'Author: {AUTHOR}')
    print(f'License: {LICENSE_NAME}')
    print('=' * 72)
    print()
    print('Portable simulator of GPU optimization logic.')
    print()
    print(f'Local:  http://localhost:{PORT}/')
    print(f'LAN:    http://{find_local_ip()}:{PORT}/')
    print(f'Logs:   {LOG_FILE}')
    print()
    print('Press Ctrl+C to stop.')
    print()

    server = ThreadedHTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping server.')
        server.server_close()
        log('Server stopped cleanly.')

if __name__ == '__main__':
    main()
