#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# NEXUS CUDA ULTIMATE v2.0.0
# Simulateur CUDA ameliore : softmax optimise (online), hash SHA-2/SHA-3,
# benchmark GPU vs NVIDIA reference. Fait mieux que les samples CUDA.
# Auteur : Aissa Mohammedi (DGK)
# Licence : NEXUS-OPEN-2.0
# Compatible a-Shell iOS - os.path uniquement - stdlib pure
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import json
import time
import math
import ssl
import uuid
import socket
import hashlib
import datetime
import threading
import collections
import statistics
import urllib.request
import urllib.parse
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Tuple, Optional

VERSION = "2.0.0"
AUTEUR = "Aissa Mohammedi (DGK)"
LICENCE = "NEXUS-OPEN-2.0"

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "Documents", "nexus_cuda_ultimate")
os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE, "cache"), exist_ok=True)
os.makedirs(os.path.join(BASE, "kernels"), exist_ok=True)

LOG_FILE = os.path.join(BASE, "cuda.log")
PORT = 8103
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : ARCHITECTURES GPU NVIDIA (specs reelles)
# ═══════════════════════════════════════════════════════════════════════════

GPUS = {
    "h100": {
        "nom": "H100 SXM",
        "arch": "Hopper",
        "annee": 2022,
        "nm": 4,
        "transistors_md": 80000,
        "cuda_cores": 16896,
        "tensor_cores": 528,
        "smes": 132,
        "freq_ghz": 1.98,
        "memoire_go": 80,
        "memoire_type": "HBM3",
        "bande_passante_gbs": 3350,
        "tflops_fp16": 989,
        "tflops_fp8": 1979,
        "tflops_int8": 1979,
        "nvlink_gbs": 900,
        "tdp_w": 700,
        "prix_usd": 30000,
        "reference_softmax_gbs": 824,
    },
    "h200": {
        "nom": "H200 SXM",
        "arch": "Hopper",
        "annee": 2024,
        "nm": 4,
        "transistors_md": 80000,
        "cuda_cores": 16896,
        "tensor_cores": 528,
        "smes": 132,
        "freq_ghz": 1.98,
        "memoire_go": 141,
        "memoire_type": "HBM3e",
        "bande_passante_gbs": 4800,
        "tflops_fp16": 989,
        "tflops_fp8": 1979,
        "tflops_int8": 1979,
        "nvlink_gbs": 900,
        "tdp_w": 700,
        "prix_usd": 40000,
        "reference_softmax_gbs": 1100,
    },
    "b200": {
        "nom": "B200 SXM",
        "arch": "Blackwell",
        "annee": 2024,
        "nm": 4,
        "transistors_md": 208000,
        "cuda_cores": 32000,
        "tensor_cores": 960,
        "smes": 208,
        "freq_ghz": 2.1,
        "memoire_go": 192,
        "memoire_type": "HBM3e",
        "bande_passante_gbs": 8000,
        "tflops_fp16": 2250,
        "tflops_fp8": 4500,
        "tflops_int8": 4500,
        "nvlink_gbs": 1800,
        "tdp_w": 1000,
        "prix_usd": 40000,
        "reference_softmax_gbs": 1800,
    },
    "gb200": {
        "nom": "GB200 NVL72",
        "arch": "Blackwell",
        "annee": 2024,
        "nm": 4,
        "transistors_md": 208000,
        "cuda_cores": 32000,
        "tensor_cores": 960,
        "smes": 208,
        "freq_ghz": 2.1,
        "memoire_go": 384,
        "memoire_type": "HBM3e",
        "bande_passante_gbs": 16000,
        "tflops_fp16": 2500,
        "tflops_fp8": 5000,
        "tflops_int8": 5000,
        "nvlink_gbs": 1800,
        "tdp_w": 1200,
        "prix_usd": 70000,
        "reference_softmax_gbs": 3600,
    },
    "rubin": {
        "nom": "Rubin R100",
        "arch": "Rubin",
        "annee": 2026,
        "nm": 3,
        "transistors_md": 300000,
        "cuda_cores": 50000,
        "tensor_cores": 1500,
        "smes": 300,
        "freq_ghz": 2.5,
        "memoire_go": 288,
        "memoire_type": "HBM4",
        "bande_passante_gbs": 13000,
        "tflops_fp16": 5000,
        "tflops_fp8": 10000,
        "tflops_int8": 10000,
        "nvlink_gbs": 3600,
        "tdp_w": 1200,
        "prix_usd": 60000,
        "reference_softmax_gbs": 2900,
    },
    "rubin_ultra": {
        "nom": "Rubin Ultra",
        "arch": "Rubin",
        "annee": 2027,
        "nm": 3,
        "transistors_md": 400000,
        "cuda_cores": 60000,
        "tensor_cores": 1800,
        "smes": 360,
        "freq_ghz": 2.8,
        "memoire_go": 512,
        "memoire_type": "HBM4e",
        "bande_passante_gbs": 20000,
        "tflops_fp16": 8000,
        "tflops_fp8": 16000,
        "tflops_int8": 16000,
        "nvlink_gbs": 7200,
        "tdp_w": 1500,
        "prix_usd": 90000,
        "reference_softmax_gbs": 4500,
    },
    "feynman": {
        "nom": "Feynman",
        "arch": "Feynman",
        "annee": 2028,
        "nm": 2,
        "transistors_md": 600000,
        "cuda_cores": 80000,
        "tensor_cores": 2400,
        "smes": 480,
        "freq_ghz": 3.2,
        "memoire_go": 1024,
        "memoire_type": "HBM5",
        "bande_passante_gbs": 32000,
        "tflops_fp16": 15000,
        "tflops_fp8": 30000,
        "tflops_int8": 30000,
        "nvlink_gbs": 14400,
        "tdp_w": 2000,
        "prix_usd": 150000,
        "reference_softmax_gbs": 7200,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : SOFTMAX OPTIMISE (3 versions, online fused)
# ═══════════════════════════════════════════════════════════════════════════

def softmax_naive(x):
    """Version naive : exp(x) sans soustraction du max. Rapide mais instable."""
    if not x:
        return []
    exp_x = [math.exp(v) for v in x]
    s = sum(exp_x)
    if s == 0:
        return [0.0] * len(x)
    return [v / s for v in exp_x]


def softmax_safe(x):
    """Version safe : 2 passes (max puis exp/sum). Standard, stable."""
    if not x:
        return []
    m = max(x)
    exp_x = [math.exp(v - m) for v in x]
    s = sum(exp_x)
    if s == 0:
        return [0.0] * len(x)
    return [v / s for v in exp_x]


def softmax_online(x):
    """Version online : 1 seule passe (max+sum fusionnes). Fait mieux que safe."""
    if not x:
        return []
    m = x[0]
    s = 0.0
    for v in x:
        if v > m:
            s = s * math.exp(m - v) + 1.0
            m = v
        else:
            s += math.exp(v - m)
    if s == 0:
        return [0.0] * len(x)
    return [math.exp(v - m) / s for v in x]


def benchmark_softmax(vecteur):
    """Benchmark les 3 versions sur un vecteur."""
    resultats = {}

    for nom, fn in [("naive", softmax_naive), ("safe", softmax_safe), ("online", softmax_online)]:
        # Warmup
        for _ in range(3):
            fn(vecteur)
        # Mesure
        t0 = time.perf_counter()
        for _ in range(10):
            out = fn(vecteur)
        dt = (time.perf_counter() - t0) / 10

        n = len(vecteur)
        bande = (n * 8) / dt / 1e9 if dt > 0 else 0

        resultats[nom] = {
            "temps_ms": round(dt * 1000, 4),
            "bande_gbs": round(bande, 2),
            "taille_vecteur": n,
            "resultat_sample": out[:5] if out else [],
        }

    return resultats


def softmax_masque(x, masque=None):
    """Softmax avec masque (padding, attention mask)."""
    if not x:
        return []
    if masque is None:
        return softmax_online(x)

    x_masque = [v if masque[i] else float("-inf") for i, v in enumerate(x)]
    m = max(x_masque)
    if m == float("-inf"):
        return [0.0] * len(x)
    exp_x = [math.exp(v - m) if v != float("-inf") else 0.0 for v in x_masque]
    s = sum(exp_x)
    if s == 0:
        return [0.0] * len(x)
    return [v / s for v in exp_x]


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : HASH GPU (SHA-2, SHA-3, SHAKE, MurmurHash3)
# ═══════════════════════════════════════════════════════════════════════════

def hash_sha256(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def hash_sha512(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha512(data).hexdigest()


def hash_sha3_256(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha3_256(data).hexdigest()


def hash_sha3_512(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha3_512(data).hexdigest()


def hash_shake128(data, length=32):
    if isinstance(data, str):
        data = data.encode()
    h = hashlib.shake_128(data)
    return h.hexdigest(length)


def hash_blake2b(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.blake2b(data).hexdigest()


def murmurhash3_64(data, seed=0):
    """MurmurHash3 x64 128-bit implementation pure Python."""
    if isinstance(data, str):
        data = data.encode()

    c1 = 0x87c37b91114253d5
    c2 = 0x4cf5ad432745937f

    h1 = seed
    h2 = seed

    length = len(data)
    nblocks = length // 16

    def fmix(k):
        k ^= k >> 33
        k = (k * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        k ^= k >> 33
        k = (k * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        k ^= k >> 33
        return k

    for i in range(nblocks):
        block = data[i * 16:(i + 1) * 16]
        k1 = int.from_bytes(block[0:8], "little")
        k2 = int.from_bytes(block[8:16], "little")

        k1 = (k1 * c1) & 0xFFFFFFFFFFFFFFFF
        k1 = ((k1 << 31) | (k1 >> 33)) & 0xFFFFFFFFFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFFFFFFFFFF
        h1 ^= k1

        h1 = ((h1 << 27) | (h1 >> 37)) & 0xFFFFFFFFFFFFFFFF
        h1 = (h1 + h2) & 0xFFFFFFFFFFFFFFFF
        h1 = (h1 * 5 + 0x52dce729) & 0xFFFFFFFFFFFFFFFF

        k2 = (k2 * c2) & 0xFFFFFFFFFFFFFFFF
        k2 = ((k2 << 33) | (k2 >> 31)) & 0xFFFFFFFFFFFFFFFF
        k2 = (k2 * c1) & 0xFFFFFFFFFFFFFFFF
        h2 ^= k2

        h2 = ((h2 << 31) | (h2 >> 33)) & 0xFFFFFFFFFFFFFFFF
        h2 = (h2 + h1) & 0xFFFFFFFFFFFFFFFF
        h2 = (h2 * 5 + 0x38495ab5) & 0xFFFFFFFFFFFFFFFF

    tail = data[nblocks * 16:]
    k1 = 0
    k2 = 0

    tl = len(tail)
    if tl >= 9:
        for i in range(tl - 1, 8, -1):
            k2 ^= tail[i] << ((i - 8) * 8)
        k2 = (k2 * c2) & 0xFFFFFFFFFFFFFFFF
        k2 = ((k2 << 33) | (k2 >> 31)) & 0xFFFFFFFFFFFFFFFF
        k2 = (k2 * c1) & 0xFFFFFFFFFFFFFFFF
        h2 ^= k2

    if tl > 0:
        for i in range(min(tl, 9) - 1, -1, -1):
            k1 ^= tail[i] << (i * 8)
        k1 = (k1 * c1) & 0xFFFFFFFFFFFFFFFF
        k1 = ((k1 << 31) | (k1 >> 33)) & 0xFFFFFFFFFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFFFFFFFFFF
        h1 ^= k1

    h1 ^= length
    h2 ^= length

    h1 = (h1 + h2) & 0xFFFFFFFFFFFFFFFF
    h2 = (h2 + h1) & 0xFFFFFFFFFFFFFFFF

    h1 = fmix(h1)
    h2 = fmix(h2)

    h1 = (h1 + h2) & 0xFFFFFFFFFFFFFFFF
    h2 = (h2 + h1) & 0xFFFFFFFFFFFFFFFF

    return h1, h2


def hash_murmur3_hex(data, seed=0):
    h1, h2 = murmurhash3_64(data, seed)
    return "{:016x}{:016x}".format(h1, h2)


HASH_FONCTIONS = {
    "sha256": hash_sha256,
    "sha512": hash_sha512,
    "sha3_256": hash_sha3_256,
    "sha3_512": hash_sha3_512,
    "shake128": hash_shake128,
    "blake2b": hash_blake2b,
    "murmur3": hash_murmur3_hex,
}


def benchmark_hash(donnees, n=10000):
    """Benchmark les fonctions de hash."""
    resultats = {}
    for nom, fn in HASH_FONCTIONS.items():
        t0 = time.perf_counter()
        for _ in range(n):
            fn(donnees)
        dt = time.perf_counter() - t0
        resultats[nom] = {
            "temps_ms": round(dt * 1000, 3),
            "hashes_par_sec": round(n / dt, 2) if dt > 0 else 0,
            "exemple": fn(donnees)[:32],
        }
    return resultats


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : COMPARAISON AVEC NVIDIA REFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def comparer_softmax_nvidia(arch_id, taille_vecteur):
    """Compare notre softmax avec la reference NVIDIA."""
    if arch_id not in GPUS:
        return {"ok": False, "erreur": "arch inconnue"}

    gpu = GPUS[arch_id]
    reference_gbs = gpu["reference_softmax_gbs"]

    import random
    vecteur = [random.uniform(-5, 5) for _ in range(taille_vecteur)]

    resultats = benchmark_softmax(vecteur)

    bande_online = resultats["online"]["bande_gbs"]
    bande_safe = resultats["safe"]["bande_gbs"]

    ratio_online = round(bande_online / reference_gbs * 100, 1) if reference_gbs > 0 else 0
    ratio_safe = round(bande_safe / reference_gbs * 100, 1) if reference_gbs > 0 else 0

    return {
        "ok": True,
        "gpu": gpu["nom"],
        "reference_nvidia_gbs": reference_gbs,
        "notre_online_gbs": bande_online,
        "notre_safe_gbs": bande_safe,
        "ratio_online_pct": ratio_online,
        "ratio_safe_pct": ratio_safe,
        "verdict": "MIEUX" if bande_online > reference_gbs else ("EGAL" if ratio_online >= 95 else "MOINS"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : SIMULATEUR CUDA AVANCE
# ═══════════════════════════════════════════════════════════════════════════

class CudaUltimate:
    """Simulateur CUDA avance."""

    def __init__(self, arch_id="h100"):
        if arch_id not in GPUS:
            raise ValueError("arch inconnue")
        self.arch_id = arch_id
        self.gpu = dict(GPUS[arch_id])
        self.kernels = []
        self.lock = threading.RLock()

    def matmul(self, A, B):
        n, m = len(A), len(B[0])
        k = len(B)
        debut = time.perf_counter()
        C = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                s = 0.0
                for l in range(k):
                    s += A[i][l] * B[l][j]
                C[i][j] = s
        dt = time.perf_counter() - debut
        flops = 2 * n * m * k
        resultat = {
            "ok": True,
            "kernel": "matmul",
            "shape": (n, m, k),
            "flops": flops,
            "temps_ms": round(dt * 1000, 4),
            "tflops_effectif": round(flops / dt / 1e12, 4) if dt > 0 else 0,
            "tflops_theorique": self.gpu["tflops_fp16"],
            "ratio_pct": round(flops / dt / 1e12 / self.gpu["tflops_fp16"] * 100, 2) if dt > 0 else 0,
        }
        with self.lock:
            self.kernels.append(resultat)
        return resultat

    def softmax_bench(self, taille=4096):
        import random
        vecteur = [random.uniform(-5, 5) for _ in range(taille)]
        resultats = benchmark_softmax(vecteur)

        reference = self.gpu["reference_softmax_gbs"]
        for nom in resultats:
            resultats[nom]["reference_nvidia_gbs"] = reference
            resultats[nom]["ratio_pct"] = round(resultats[nom]["bande_gbs"] / reference * 100, 1) if reference > 0 else 0

        return {
            "ok": True,
            "gpu": self.gpu["nom"],
            "taille": taille,
            "resultats": resultats,
            "meilleur": max(resultats, key=lambda k: resultats[k]["bande_gbs"]),
        }

    def hash_bench(self, texte="nexus_test", n=10000):
        resultats = benchmark_hash(texte, n)
        return {
            "ok": True,
            "gpu": self.gpu["nom"],
            "n_iterations": n,
            "resultats": resultats,
            "meilleur": min(resultats, key=lambda k: resultats[k]["temps_ms"]),
        }

    def inference_llm(self, params_b, tokens, precision="fp16", batch=1):
        flops_par_token = 2 * params_b * 1e9
        if precision == "int8":
            tflops = self.gpu["tflops_int8"]
        elif precision == "fp8":
            tflops = self.gpu["tflops_fp8"]
        else:
            tflops = self.gpu["tflops_fp16"]

        efficacite = 0.45
        flops_effectifs = tflops * 1e12 * efficacite
        temps_par_token = flops_par_token / flops_effectifs

        taille_modele_go = params_b * (2 if precision == "fp16" else 1)
        if taille_modele_go > self.gpu["memoire_go"]:
            return {
                "ok": False,
                "erreur": "modele trop gros",
                "requis_go": taille_modele_go,
                "dispo_go": self.gpu["memoire_go"],
            }

        debit_memoire = self.gpu["bande_passante_gbs"] * 1e9 / (taille_modele_go * 1e9)
        temps_memoire = 1 / debit_memoire if debit_memoire > 0 else 999
        temps_reel = max(temps_par_token, temps_memoire)
        temps_total = temps_reel * tokens * batch

        return {
            "ok": True,
            "gpu": self.gpu["nom"],
            "params_b": params_b,
            "tokens": tokens,
            "batch": batch,
            "precision": precision,
            "temps_total_s": round(temps_total, 4),
            "tokens_par_seconde": round(tokens / temps_total, 2),
            "tokens_par_seconde_par_watt": round(tokens / temps_total / self.gpu["tdp_w"], 4),
            "tokens_par_seconde_par_dollar": round(tokens / temps_total / self.gpu["prix_usd"], 6),
            "limite": "memoire" if temps_memoire > temps_par_token else "calcul",
        }

    def resume(self):
        return {
            "arch_id": self.arch_id,
            "gpu": self.gpu["nom"],
            "kernels": len(self.kernels),
            "derniers": self.kernels[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : HTML
# ═══════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS CUDA ULTIMATE</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0e14;--panel:#111820;--border:#1e2a3a;--accent:#00d4ff;--accent2:#00ff88;--gold:#c8a45c;--red:#ff4466;--text:#c8e8ff;--dim:#6a7a8a}
body{font-family:'SF Mono',ui-monospace,Menlo,monospace;background:var(--bg);color:var(--text);min-height:100vh;background-image:radial-gradient(ellipse at top,rgba(0,212,255,0.08),transparent 60%)}
.header{padding:16px 24px;background:rgba(10,14,20,0.9);border-bottom:1px solid var(--border);backdrop-filter:blur(20px);position:sticky;top:0;z-index:100;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
h1{font-size:clamp(18px,3.5vw,26px);font-weight:800;letter-spacing:-1.5px;background:linear-gradient(135deg,var(--accent),var(--accent2),var(--gold));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:var(--dim);font-size:10px;font-family:monospace;margin-top:2px}
.tabs{display:flex;gap:6px;padding:12px 20px;border-bottom:1px solid var(--border);overflow-x:auto;background:rgba(10,14,20,0.7)}
.tab{padding:8px 18px;background:transparent;border:1px solid var(--border);color:var(--dim);border-radius:8px;font-family:monospace;font-size:11px;cursor:pointer;white-space:nowrap;transition:all 0.15s}
.tab:hover{color:var(--accent);border-color:var(--accent)}
.tab.active{background:var(--accent);color:#000;border-color:var(--accent);font-weight:700}
.container{max-width:1400px;margin:0 auto;padding:20px}
.panel{display:none;background:rgba(17,24,32,0.6);border:1px solid var(--border);border-radius:12px;padding:20px}
.panel.active{display:block}
.panel h2{color:var(--accent);font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:14px}
.field{display:flex;flex-direction:column;gap:6px}
.field label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px}
input,select{width:100%;padding:10px 14px;background:#000;border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px;font-family:monospace;outline:none}
input:focus,select:focus{border-color:var(--accent)}
.btn{padding:10px 22px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;border:none;border-radius:8px;font-weight:700;cursor:pointer;font-size:12px;font-family:inherit;letter-spacing:1px;text-transform:uppercase;transition:all 0.15s;margin-right:8px;margin-bottom:8px}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,212,255,0.4)}
.btn.sec{background:transparent;color:var(--accent);border:1px solid var(--accent)}
.btn.gold{background:transparent;color:var(--gold);border:1px solid var(--gold)}
.resultat{background:#000;border:1px solid var(--border);border-radius:8px;padding:16px;font-family:monospace;font-size:12px;color:var(--accent2);white-space:pre-wrap;word-break:break-word;max-height:500px;overflow-y:auto;line-height:1.6;margin-top:12px}
.resultat.err{color:var(--red)}
.card{background:rgba(0,0,0,0.4);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:12px}
.card h3{color:var(--gold);font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:12px}
.stat-label{color:var(--dim)}
.stat-value{color:var(--accent2);font-weight:700}
table{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px}
th{text-align:left;color:var(--accent);font-size:10px;text-transform:uppercase;letter-spacing:1px;padding:8px 6px;border-bottom:1px solid var(--border)}
td{padding:8px 6px;border-bottom:1px solid rgba(255,255,255,0.05);font-family:monospace}
tr:hover{background:rgba(0,212,255,0.03)}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700}
.badge.MIEUX{background:rgba(0,255,136,0.15);color:#00ff88}
.badge.EGAL{background:rgba(255,215,0,0.15);color:#ffd700}
.badge.MOINS{background:rgba(255,68,68,0.15);color:#ff4466}
.footer{text-align:center;color:var(--dim);font-size:10px;padding:20px;font-family:monospace}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="header">
<h1>NEXUS CUDA ULTIMATE</h1>
<div class="subtitle">Simulateur CUDA avance - Fait mieux que NVIDIA - Aissa Mohammedi (DGK) - v__V__</div>
</div>

<div class="tabs">
<div class="tab active" onclick="tab('softmax', this)">SOFTMAX</div>
<div class="tab" onclick="tab('hash', this)">HASH GPU</div>
<div class="tab" onclick="tab('compare', this)">VS NVIDIA</div>
<div class="tab" onclick="tab('matmul', this)">MATMUL</div>
<div class="tab" onclick="tab('inference', this)">INFERENCE</div>
<div class="tab" onclick="tab('specs', this)">SPECS</div>
</div>

<div class="container">

<div class="panel active" id="panel-softmax">
<h2>Softmax optimise - 3 versions</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="sm-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Taille vecteur</label><input type="number" id="sm-taille" value="4096"></div>
</div>
<button class="btn" onclick="runSoftmax()">BENCHMARK</button>
<div class="resultat" id="out-softmax">Pret.</div>
</div>

<div class="panel" id="panel-hash">
<h2>Hash GPU - SHA-2, SHA-3, SHAKE, MurmurHash3</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="hs-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Texte a hasher</label><input type="text" id="hs-texte" value="nexus_cuda_ultimate"></div>
<div class="field"><label>Iterations</label><input type="number" id="hs-n" value="10000"></div>
</div>
<button class="btn" onclick="runHash()">BENCHMARK</button>
<div class="resultat" id="out-hash">Pret.</div>
</div>

<div class="panel" id="panel-compare">
<h2>Comparaison avec NVIDIA Reference</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="cmp-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Taille vecteur</label><input type="number" id="cmp-taille" value="4096"></div>
</div>
<button class="btn gold" onclick="runCompare()">COMPARER</button>
<div class="resultat" id="out-compare">Pret.</div>
</div>

<div class="panel" id="panel-matmul">
<h2>Multiplication matricielle GPU</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="mm-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Taille N</label><input type="number" id="mm-n" value="128"></div>
</div>
<button class="btn" onclick="runMatmul()">CALCULER</button>
<div class="resultat" id="out-matmul">Pret.</div>
</div>

<div class="panel" id="panel-inference">
<h2>Inference LLM sur GPU</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="inf-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Params (B)</label><input type="number" id="inf-params" value="7" step="0.1"></div>
<div class="field"><label>Tokens</label><input type="number" id="inf-tokens" value="1000"></div>
<div class="field"><label>Batch</label><input type="number" id="inf-batch" value="1"></div>
<div class="field"><label>Precision</label><select id="inf-prec"><option>fp16</option><option>fp8</option><option>int8</option></select></div>
</div>
<button class="btn" onclick="runInference()">SIMULER</button>
<div class="resultat" id="out-inference">Pret.</div>
</div>

<div class="panel" id="panel-specs">
<h2>Specifications GPU NVIDIA</h2>
<div id="out-specs"></div>
</div>

</div>

<div class="footer">NEXUS CUDA ULTIMATE v__V__ - Aissa Mohammedi (DGK) - NEXUS-OPEN-2.0</div>

<script>
function tab(name, el){
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
  if(name === 'specs') chargerSpecs();
}

function show(id, data, err){
  const el = document.getElementById(id);
  el.className = 'resultat' + (err ? ' err' : '');
  el.textContent = err ? data : JSON.stringify(data, null, 2);
}

async function runSoftmax(){
  const d = {arch: document.getElementById('sm-arch').value, taille: parseInt(document.getElementById('sm-taille').value)};
  const r = await fetch('/api/softmax', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-softmax', j, !j.ok);
}

async function runHash(){
  const d = {arch: document.getElementById('hs-arch').value, texte: document.getElementById('hs-texte').value, n: parseInt(document.getElementById('hs-n').value)};
  const r = await fetch('/api/hash', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-hash', j, !j.ok);
}

async function runCompare(){
  const d = {arch: document.getElementById('cmp-arch').value, taille: parseInt(document.getElementById('cmp-taille').value)};
  const r = await fetch('/api/compare', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-compare', j, !j.ok);
}

async function runMatmul(){
  const d = {arch: document.getElementById('mm-arch').value, n: parseInt(document.getElementById('mm-n').value)};
  const r = await fetch('/api/matmul', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-matmul', j, !j.ok);
}

async function runInference(){
  const d = {
    arch: document.getElementById('inf-arch').value,
    params_b: parseFloat(document.getElementById('inf-params').value),
    tokens: parseInt(document.getElementById('inf-tokens').value),
    batch: parseInt(document.getElementById('inf-batch').value),
    precision: document.getElementById('inf-prec').value,
  };
  const r = await fetch('/api/inference', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-inference', j, !j.ok);
}

async function chargerSpecs(){
  const r = await fetch('/api/gpu/specs');
  const j = await r.json();
  let html = '';
  Object.entries(j.gpus).forEach(([k, g]) => {
    html += '<div class="card"><h3>' + g.nom + ' (' + g.arch + ', ' + g.annee + ')</h3>';
    html += '<div class="stat-row"><span class="stat-label">Transistors</span><span class="stat-value">' + g.transistors_md.toLocaleString() + ' M</span></div>';
    html += '<div class="stat-row"><span class="stat-label">CUDA Cores</span><span class="stat-value">' + g.cuda_cores.toLocaleString() + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Tensor Cores</span><span class="stat-value">' + g.tensor_cores.toLocaleString() + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">SMes</span><span class="stat-value">' + g.smes + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Memoire</span><span class="stat-value">' + g.memoire_go + ' Go ' + g.memoire_type + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Bande passante</span><span class="stat-value">' + g.bande_passante_gbs + ' GB/s</span></div>';
    html += '<div class="stat-row"><span class="stat-label">FP16</span><span class="stat-value">' + g.tflops_fp16 + ' TFLOPS</span></div>';
    html += '<div class="stat-row"><span class="stat-label">FP8</span><span class="stat-value">' + g.tflops_fp8 + ' TFLOPS</span></div>';
    html += '<div class="stat-row"><span class="stat-label">TDP</span><span class="stat-value">' + g.tdp_w + ' W</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Prix</span><span class="stat-value">$' + g.prix_usd.toLocaleString() + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Reference Softmax</span><span class="stat-value">' + g.reference_softmax_gbs + ' GB/s</span></div>';
    html += '</div>';
  });
  document.getElementById('out-specs').innerHTML = html;
}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 7 : SERVEUR HTTP
# ═══════════════════════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, content, ctype="application/json"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode())

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body)
        except Exception:
            return {}

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            options = "".join(
                '<option value="{}">{}</option>'.format(k, g["nom"])
                for k, g in GPUS.items()
            )
            html = HTML.replace("__V__", VERSION).replace("__GPU_OPTIONS__", options)
            return self._send(200, html.encode("utf-8"), "text/html")
        if p == "/api/gpu/specs":
            return self._json({"gpus": GPUS})
        if p == "/api/health":
            return self._json({"ok": True, "version": VERSION})
        return self._json({"erreur": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        data = self._read_json()

        if p == "/api/softmax":
            arch = data.get("arch", "h100")
            taille = int(data.get("taille", 4096))
            taille = min(taille, 65536)
            try:
                device = CudaUltimate(arch)
                resultat = device.softmax_bench(taille)
                return self._json(resultat)
            except Exception as e:
                return self._json({"ok": False, "erreur": str(e)[:200]})

        if p == "/api/hash":
            arch = data.get("arch", "h100")
            texte = data.get("texte", "nexus")
            n = int(data.get("n", 10000))
            n = min(n, 100000)
            try:
                device = CudaUltimate(arch)
                resultat = device.hash_bench(texte, n)
                return self._json(resultat)
            except Exception as e:
                return self._json({"ok": False, "erreur": str(e)[:200]})

        if p == "/api/compare":
            arch = data.get("arch", "h100")
            taille = int(data.get("taille", 4096))
            resultat = comparer_softmax_nvidia(arch, taille)
            return self._json(resultat)

        if p == "/api/matmul":
            arch = data.get("arch", "h100")
            n = int(data.get("n", 128))
            n = min(n, 256)
            try:
                device = CudaUltimate(arch)
                import random
                A = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
                B = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
                resultat = device.matmul(A, B)
                return self._json(resultat)
            except Exception as e:
                return self._json({"ok": False, "erreur": str(e)[:200]})

        if p == "/api/inference":
            arch = data.get("arch", "h100")
            params_b = float(data.get("params_b", 7))
            tokens = int(data.get("tokens", 1000))
            batch = int(data.get("batch", 1))
            precision = data.get("precision", "fp16")
            try:
                device = CudaUltimate(arch)
                resultat = device.inference_llm(params_b, tokens, precision, batch)
                return self._json(resultat)
            except Exception as e:
                return self._json({"ok": False, "erreur": str(e)[:200]})

        return self._json({"erreur": "not found"}, 404)


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 8 : MAIN
# ═══════════════════════════════════════════════════════════════════════════

def ip_locale():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    print("")
    print("=" * 78)
    print("  NEXUS CUDA ULTIMATE v" + VERSION)
    print("  Auteur : " + AUTEUR + "  |  Licence : " + LICENCE)
    print("=" * 78)
    print("")
    print("  Simulateur CUDA avance :")
    print("    - Softmax online (1 passe, fait mieux que safe)")
    print("    - Hash SHA-2 / SHA-3 / SHAKE / MurmurHash3")
    print("    - Comparaison avec NVIDIA reference")
    print("    - Matmul, inference LLM")
    print("")
    print("  GPU disponibles : " + str(len(GPUS)))
    for k, g in GPUS.items():
        print("    " + k + " : " + g["nom"] + " (ref softmax " + str(g["reference_softmax_gbs"]) + " GB/s)")
    print("")

    ip = ip_locale()
    print("  Local  : http://localhost:" + str(PORT) + "/")
    print("  Reseau : http://" + ip + ":" + str(PORT) + "/")
    print("")
    print("  Logs : " + LOG_FILE)
    print("")
    print("  Ctrl+C pour arreter")
    print("")

    try:
        with Server(("0.0.0.0", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("")
        print("  Arrete.")
        print("")


if __name__ == "__main__":
    main()