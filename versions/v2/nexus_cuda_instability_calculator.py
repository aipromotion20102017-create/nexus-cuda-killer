#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# NEXUS CUDA SPARK MUSE v1.0.0
# Simulateur de calcul GPU. Reimplementation des primitives CUDA en Python.
# Calcule comme si tu avais un H100/B200/Rubin avec Spark + Muse.
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

VERSION = "1.0.0"
AUTEUR = "Aissa Mohammedi (DGK)"
LICENCE = "NEXUS-OPEN-2.0"

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "Documents", "nexus_cuda_spark_muse")
os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE, "cache"), exist_ok=True)

LOG_FILE = os.path.join(BASE, "cuda.log")
PORT = 8102
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
# PARTIE 1 : ARCHITECTURES GPU NVIDIA
# ═══════════════════════════════════════════════════════════════════════════

GPUS = {
    "h100": {
        "nom": "H100 Hopper",
        "annee": 2022,
        "architecture": "Hopper",
        "process_nm": 4,
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
        "tflops_fp32": 67,
        "nvlink_gbs": 900,
        "tdp_w": 700,
        "prix_usd": 30000,
    },
    "h200": {
        "nom": "H200 Hopper",
        "annee": 2024,
        "architecture": "Hopper",
        "process_nm": 4,
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
        "tflops_fp32": 67,
        "nvlink_gbs": 900,
        "tdp_w": 700,
        "prix_usd": 40000,
    },
    "b200": {
        "nom": "B200 Blackwell",
        "annee": 2024,
        "architecture": "Blackwell",
        "process_nm": 4,
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
        "tflops_fp32": 80,
        "nvlink_gbs": 1800,
        "tdp_w": 1000,
        "prix_usd": 40000,
    },
    "gb200": {
        "nom": "GB200 Grace Blackwell",
        "annee": 2024,
        "architecture": "Blackwell",
        "process_nm": 4,
        "transistors_md": 208000,
        "cuda_cores": 32000,
        "tensor_cores": 960,
        "smes": 208,
        "freq_ghz": 2.1,
        "memoire_go": 384,
        "memoire_type": "HBM3e + LPDDR5X",
        "bande_passante_gbs": 16000,
        "tflops_fp16": 2500,
        "tflops_fp8": 5000,
        "tflops_int8": 5000,
        "tflops_fp32": 90,
        "nvlink_gbs": 1800,
        "tdp_w": 1200,
        "prix_usd": 70000,
    },
    "rubin": {
        "nom": "Rubin R100",
        "annee": 2026,
        "architecture": "Rubin",
        "process_nm": 3,
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
        "tflops_fp32": 120,
        "nvlink_gbs": 3600,
        "tdp_w": 1200,
        "prix_usd": 60000,
    },
    "rubin_ultra": {
        "nom": "Rubin Ultra",
        "annee": 2027,
        "architecture": "Rubin",
        "process_nm": 3,
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
        "tflops_fp32": 180,
        "nvlink_gbs": 7200,
        "tdp_w": 1500,
        "prix_usd": 90000,
    },
    "feynman": {
        "nom": "Feynman",
        "annee": 2028,
        "architecture": "Feynman",
        "process_nm": 2,
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
        "tflops_fp32": 300,
        "nvlink_gbs": 14400,
        "tdp_w": 2000,
        "prix_usd": 150000,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : SIMULATEUR CUDA (primitives GPU en Python)
# ═══════════════════════════════════════════════════════════════════════════

class CudaKernel:
    """Represente un kernel CUDA."""

    def __init__(self, nom, nb_threads, nb_blocks):
        self.nom = nom
        self.nb_threads = nb_threads
        self.nb_blocks = nb_blocks
        self.temps_debut = None
        self.temps_fin = None

    def __enter__(self):
        self.temps_debut = time.time()
        return self

    def __exit__(self, *args):
        self.temps_fin = time.time()

    @property
    def duree_ms(self):
        if self.temps_debut and self.temps_fin:
            return round((self.temps_fin - self.temps_debut) * 1000, 3)
        return 0


class CudaDevice:
    """Simulateur d'un device CUDA."""

    def __init__(self, arch_id):
        if arch_id not in GPUS:
            raise ValueError("architecture inconnue : " + arch_id)
        self.arch_id = arch_id
        self.gpu = dict(GPUS[arch_id])
        self.kernels = []
        self.memoire_allouee_go = 0
        self.lock = threading.RLock()

    def allouer(self, taille_go):
        with self.lock:
            if self.memoire_allouee_go + taille_go > self.gpu["memoire_go"]:
                return {"ok": False, "erreur": "OOM", "dispo_go": self.gpu["memoire_go"] - self.memoire_allouee_go}
            self.memoire_allouee_go += taille_go
            return {"ok": True, "alloue_go": taille_go, "total_alloue_go": self.memoire_allouee_go}

    def liberer(self, taille_go):
        with self.lock:
            self.memoire_allouee_go = max(0, self.memoire_allouee_go - taille_go)

    def kernel(self, nom, nb_threads=1024, nb_blocks=None):
        if nb_blocks is None:
            nb_blocks = self.gpu["smes"] * 4
        return CudaKernel(nom, nb_threads, nb_blocks)

    def matmul(self, A, B):
        """Multiplication matricielle simulee sur GPU."""
        n, m = len(A), len(B[0])
        k = len(B)
        with self.kernel("matmul", nb_threads=256, nb_blocks=n * m // 256 + 1) as k_obj:
            C = [[0.0] * m for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for l in range(k):
                        s += A[i][l] * B[l][j]
                    C[i][j] = s
        resultat = {
            "ok": True,
            "kernel": "matmul",
            "shape": (n, m, k),
            "flops": 2 * n * m * k,
            "temps_ms": k_obj.duree_ms,
            "tflops_effectif": round(2 * n * m * k / (k_obj.duree_ms / 1000) / 1e12, 4) if k_obj.duree_ms > 0 else 0,
            "memoire_go": (n * k + k * m + n * m) * 8 / 1e9,
        }
        with self.lock:
            self.kernels.append(resultat)
        return resultat

    def transpose(self, M):
        n, m = len(M), len(M[0])
        with self.kernel("transpose", nb_threads=32 * 32, nb_blocks=(n * m) // 1024 + 1) as k_obj:
            R = [[M[i][j] for i in range(n)] for j in range(m)]
        resultat = {
            "ok": True,
            "kernel": "transpose",
            "shape": (n, m),
            "temps_ms": k_obj.duree_ms,
        }
        with self.lock:
            self.kernels.append(resultat)
        return resultat

    def softmax(self, vecteur):
        with self.kernel("softmax", nb_threads=256, nb_blocks=1) as k_obj:
            max_v = max(vecteur)
            exp_v = [math.exp(x - max_v) for x in vecteur]
            somme = sum(exp_v)
            resultat_vals = [x / somme for x in exp_v]
        resultat = {
            "ok": True,
            "kernel": "softmax",
            "n": len(vecteur),
            "temps_ms": k_obj.duree_ms,
            "resultat": resultat_vals[:10],
        }
        with self.lock:
            self.kernels.append(resultat)
        return resultat

    def reduction_sum(self, vecteur):
        with self.kernel("reduce_sum", nb_threads=256, nb_blocks=len(vecteur) // 256 + 1) as k_obj:
            s = sum(vecteur)
        resultat = {
            "ok": True,
            "kernel": "reduce_sum",
            "n": len(vecteur),
            "somme": s,
            "temps_ms": k_obj.duree_ms,
        }
        with self.lock:
            self.kernels.append(resultat)
        return resultat

    def inference_llm(self, params_b, tokens, precision="fp16", batch=1):
        """Simule une inference LLM sur GPU."""
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
            return {"ok": False, "erreur": "modele trop gros", "requis_go": taille_modele_go, "dispo_go": self.gpu["memoire_go"]}

        debit_memoire = self.gpu["bande_passante_gbs"] * 1e9 / (taille_modele_go * 1e9)
        temps_memoire = 1 / debit_memoire if debit_memoire > 0 else 999

        temps_par_token_reel = max(temps_par_token, temps_memoire)
        temps_total = temps_par_token_reel * tokens * batch

        return {
            "ok": True,
            "gpu": self.gpu["nom"],
            "params_b": params_b,
            "tokens": tokens,
            "batch": batch,
            "precision": precision,
            "temps_total_s": round(temps_total, 4),
            "temps_par_token_ms": round(temps_par_token_reel * 1000, 3),
            "tokens_par_seconde": round(tokens / temps_total, 2),
            "tokens_par_seconde_par_watt": round(tokens / temps_total / self.gpu["tdp_w"], 4),
            "tokens_par_seconde_par_dollar": round(tokens / temps_total / self.gpu["prix_usd"], 6),
            "limite": "memoire" if temps_memoire > temps_par_token else "calcul",
        }

    def entrainement_step(self, params_b, batch_size, seq_len, precision="fp16"):
        """Simule un step d'entrainement."""
        flops_forward = 2 * params_b * 1e9 * batch_size * seq_len
        flops_backward = 4 * params_b * 1e9 * batch_size * seq_len
        flops_total = flops_forward + flops_backward

        tflops = self.gpu["tflops_fp16"] if precision == "fp16" else self.gpu["tflops_fp8"]
        efficacite = 0.35
        flops_effectifs = tflops * 1e12 * efficacite
        temps_step = flops_total / flops_effectifs

        memoire_activations_go = batch_size * seq_len * 4096 * 4 / 1e9

        return {
            "ok": True,
            "gpu": self.gpu["nom"],
            "params_b": params_b,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "flops_total": flops_total,
            "flops_forward": flops_forward,
            "flops_backward": flops_backward,
            "temps_step_s": round(temps_step, 4),
            "steps_par_seconde": round(1 / temps_step, 2),
            "memoire_activations_go": round(memoire_activations_go, 2),
            "tflops_effectif": round(flops_total / temps_step / 1e12, 2),
        }

    def resume(self):
        return {
            "arch_id": self.arch_id,
            "gpu": self.gpu["nom"],
            "kernels_executes": len(self.kernels),
            "memoire_allouee_go": self.memoire_allouee_go,
            "derniers_kernels": self.kernels[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : SPARK — TRAITEMENT DISTRIBUE
# ═══════════════════════════════════════════════════════════════════════════

class SparkContext:
    """Simulateur minimal de Spark."""

    def __init__(self, nom, nb_partitions=8):
        self.nom = nom
        self.nb_partitions = nb_partitions
        self.rdds = []
        self.jobs = []
        self.lock = threading.RLock()

    def parallelize(self, data):
        """Cree un RDD a partir d'une liste."""
        rdd = {
            "type": "parallelize",
            "data": data,
            "partitions": self.nb_partitions,
            "size": len(data),
        }
        with self.lock:
            self.rdds.append(rdd)
        return rdd

    def map(self, rdd, fonction_nom, fonction):
        """Map distribue."""
        with self.lock:
            debut = time.time()
            resultat = [fonction(x) for x in rdd["data"]]
            duree = time.time() - debut
            new_rdd = {
                "type": "map",
                "fonction": fonction_nom,
                "data": resultat,
                "size": len(resultat),
                "temps_ms": round(duree * 1000, 3),
            }
            self.rdds.append(new_rdd)
            self.jobs.append({"op": "map", "fonction": fonction_nom, "duree_ms": new_rdd["temps_ms"]})
            return new_rdd

    def filter(self, rdd, predicat_nom, predicat):
        with self.lock:
            debut = time.time()
            resultat = [x for x in rdd["data"] if predicat(x)]
            duree = time.time() - debut
            new_rdd = {
                "type": "filter",
                "predicat": predicat_nom,
                "data": resultat,
                "size": len(resultat),
                "temps_ms": round(duree * 1000, 3),
            }
            self.rdds.append(new_rdd)
            self.jobs.append({"op": "filter", "predicat": predicat_nom, "duree_ms": new_rdd["temps_ms"]})
            return new_rdd

    def reduce(self, rdd, fonction, initial):
        with self.lock:
            debut = time.time()
            acc = initial
            for x in rdd["data"]:
                acc = fonction(acc, x)
            duree = time.time() - debut
            self.jobs.append({"op": "reduce", "duree_ms": round(duree * 1000, 3)})
            return {"ok": True, "resultat": acc, "temps_ms": round(duree * 1000, 3)}

    def count(self, rdd):
        return {"ok": True, "count": len(rdd["data"])}

    def collect(self, rdd):
        return {"ok": True, "data": rdd["data"][:100]}

    def groupByKey(self, rdd):
        """Group by key sur une liste de tuples."""
        with self.lock:
            debut = time.time()
            groupes = collections.defaultdict(list)
            for k, v in rdd["data"]:
                groupes[k].append(v)
            resultat = dict(groupes)
            duree = time.time() - debut
            new_rdd = {
                "type": "groupByKey",
                "data": list(resultat.items()),
                "size": len(resultat),
                "temps_ms": round(duree * 1000, 3),
            }
            self.rdds.append(new_rdd)
            return new_rdd

    def wordCount(self, texte):
        """Exemple classique : word count distribue."""
        with self.lock:
            debut = time.time()
            mots = re.findall(r"\w+", texte.lower())
            compteur = collections.Counter(mots)
            resultat = compteur.most_common(50)
            duree = time.time() - debut
            self.jobs.append({"op": "wordCount", "mots": len(mots), "uniques": len(compteur), "duree_ms": round(duree * 1000, 3)})
            return {
                "ok": True,
                "mots_total": len(mots),
                "mots_uniques": len(compteur),
                "top_50": resultat,
                "temps_ms": round(duree * 1000, 3),
            }

    def resume(self):
        return {
            "nom": self.nom,
            "rdds_crees": len(self.rdds),
            "jobs_executes": len(self.jobs),
            "derniers_jobs": self.jobs[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : MUSE — MOTEUR DE RECHERCHE MATRICIEL
# ═══════════════════════════════════════════════════════════════════════════

class MuseEngine:
    """Moteur de calcul matriciel optimise type MUSE."""

    def __init__(self, dim=512):
        self.dim = dim
        self.index = {}
        self.vecteurs = []
        self.metadonnees = []
        self.lock = threading.RLock()

    def encoder(self, texte):
        """Encode un texte en vecteur dense (bag-of-words hache)."""
        vec = [0.0] * self.dim
        mots = re.findall(r"\w+", texte.lower())
        for mot in mots:
            h = int(hashlib.sha256(mot.encode()).hexdigest(), 16)
            idx = h % self.dim
            signe = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += signe
        norme = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norme for x in vec]

    def ajouter(self, texte, metadata=None):
        with self.lock:
            vec = self.encoder(texte)
            idx = len(self.vecteurs)
            self.vecteurs.append(vec)
            self.metadonnees.append({"texte": texte, "metadata": metadata or {}})
            return {"ok": True, "index": idx, "dim": self.dim}

    def similarite_cosinus(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1)) or 1.0
        n2 = math.sqrt(sum(b * b for b in v2)) or 1.0
        return dot / (n1 * n2)

    def rechercher(self, requete, top_k=5):
        with self.lock:
            if not self.vecteurs:
                return {"ok": False, "erreur": "index vide"}
            vec_q = self.encoder(requete)
            scores = []
            for i, v in enumerate(self.vecteurs):
                s = self.similarite_cosinus(vec_q, v)
                scores.append((i, s, self.metadonnees[i]))
            scores.sort(key=lambda x: -x[1])
            return {
                "ok": True,
                "requete": requete,
                "resultats": [
                    {"index": i, "score": round(s, 6), "texte": m["texte"][:200], "metadata": m["metadata"]}
                    for i, s, m in scores[:top_k]
                ],
            }

    def produit_matriciel_batch(self, matrices):
        """Calcule plusieurs produits matriciels en parallele."""
        with self.lock:
            debut = time.time()
            resultats = []
            for A, B in matrices:
                n, m = len(A), len(B[0])
                k = len(B)
                C = [[0.0] * m for _ in range(n)]
                for i in range(n):
                    for j in range(m):
                        s = 0.0
                        for l in range(k):
                            s += A[i][l] * B[l][j]
                        C[i][j] = s
                resultats.append({"shape": (n, m), "ok": True})
            duree = time.time() - debut
            return {
                "ok": True,
                "batch_size": len(matrices),
                "temps_ms": round(duree * 1000, 3),
                "resultats": resultats,
            }

    def resume(self):
        with self.lock:
            return {
                "dim": self.dim,
                "vecteurs_indexes": len(self.vecteurs),
            }


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : ETAT GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

ETAT = {
    "device": None,
    "spark": None,
    "muse": None,
    "historique": [],
}
ETAT_LOCK = threading.RLock()


def get_device(arch_id="h100"):
    with ETAT_LOCK:
        if ETAT["device"] is None or ETAT["device"].arch_id != arch_id:
            ETAT["device"] = CudaDevice(arch_id)
        return ETAT["device"]


def get_spark():
    with ETAT_LOCK:
        if ETAT["spark"] is None:
            ETAT["spark"] = SparkContext("NEXUS-SPARK", nb_partitions=8)
        return ETAT["spark"]


def get_muse():
    with ETAT_LOCK:
        if ETAT["muse"] is None:
            ETAT["muse"] = MuseEngine(dim=512)
        return ETAT["muse"]


def ajouter_historique(type_calc, entree, sortie):
    evt = {
        "id": str(uuid.uuid4())[:8],
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": type_calc,
        "entree": entree,
        "sortie": sortie,
    }
    with ETAT_LOCK:
        ETAT["historique"].append(evt)
        if len(ETAT["historique"]) > 200:
            ETAT["historique"] = ETAT["historique"][-200:]


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : HTML
# ═══════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS CUDA SPARK MUSE</title>
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
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:14px}
.field{display:flex;flex-direction:column;gap:6px}
.field label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px}
input,select,textarea{width:100%;padding:10px 14px;background:#000;border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px;font-family:monospace;outline:none}
input:focus,select:focus,textarea:focus{border-color:var(--accent)}
textarea{min-height:100px;resize:vertical;line-height:1.5}
.btn{padding:10px 22px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;border:none;border-radius:8px;font-weight:700;cursor:pointer;font-size:12px;font-family:inherit;letter-spacing:1px;text-transform:uppercase;transition:all 0.15s;margin-right:8px;margin-bottom:8px}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,212,255,0.4)}
.btn.sec{background:transparent;color:var(--accent);border:1px solid var(--accent)}
.btn.sec:hover{background:rgba(0,212,255,0.1)}
.btn.gold{background:transparent;color:var(--gold);border:1px solid var(--gold)}
.btn.gold:hover{background:rgba(200,164,92,0.1)}
.resultat{background:#000;border:1px solid var(--border);border-radius:8px;padding:16px;font-family:monospace;font-size:12px;color:var(--accent2);white-space:pre-wrap;word-break:break-word;max-height:500px;overflow-y:auto;line-height:1.6;margin-top:12px}
.resultat.err{color:var(--red)}
.card{background:rgba(0,0,0,0.4);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:12px}
.card h3{color:var(--gold);font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:12px}
.stat-row:last-child{border:none}
.stat-label{color:var(--dim)}
.stat-value{color:var(--accent2);font-weight:700}
table{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px}
th{text-align:left;color:var(--accent);font-size:10px;text-transform:uppercase;letter-spacing:1px;padding:8px 6px;border-bottom:1px solid var(--border)}
td{padding:8px 6px;border-bottom:1px solid rgba(255,255,255,0.05);font-family:monospace}
tr:hover{background:rgba(0,212,255,0.03)}
.footer{text-align:center;color:var(--dim);font-size:10px;padding:20px;font-family:monospace}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="header">
<h1>NEXUS CUDA SPARK MUSE</h1>
<div class="subtitle">Simulateur GPU + Spark + MUSE - Aissa Mohammedi (DGK) - v__V__</div>
</div>

<div class="tabs">
<div class="tab active" onclick="tab('gpu', this)">GPU INFERENCE</div>
<div class="tab" onclick="tab('train', this)">ENTRAINEMENT</div>
<div class="tab" onclick="tab('matmul', this)">MATMUL</div>
<div class="tab" onclick="tab('spark', this)">SPARK</div>
<div class="tab" onclick="tab('muse', this)">MUSE RECHERCHE</div>
<div class="tab" onclick="tab('specs', this)">SPECS GPU</div>
<div class="tab" onclick="tab('bench', this)">BENCHMARK</div>
</div>

<div class="container">

<!-- TAB GPU INFERENCE -->
<div class="panel active" id="panel-gpu">
<h2>Inference LLM sur GPU</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="gpu-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Parametres (B)</label><input type="number" id="gpu-params" value="7" step="0.1"></div>
<div class="field"><label>Tokens</label><input type="number" id="gpu-tokens" value="1000"></div>
<div class="field"><label>Batch</label><input type="number" id="gpu-batch" value="1"></div>
<div class="field"><label>Precision</label><select id="gpu-prec"><option>fp16</option><option>fp8</option><option>int8</option></select></div>
</div>
<button class="btn" onclick="runGpu()">SIMULER</button>
<div class="resultat" id="out-gpu">Pret.</div>
</div>

<!-- TAB ENTRAINEMENT -->
<div class="panel" id="panel-train">
<h2>Simulation entrainement</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="tr-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Parametres (B)</label><input type="number" id="tr-params" value="7" step="0.1"></div>
<div class="field"><label>Batch size</label><input type="number" id="tr-batch" value="8"></div>
<div class="field"><label>Seq len</label><input type="number" id="tr-seq" value="4096"></div>
<div class="field"><label>Precision</label><select id="tr-prec"><option>fp16</option><option>fp8</option></select></div>
</div>
<button class="btn" onclick="runTrain()">SIMULER STEP</button>
<div class="resultat" id="out-train">Pret.</div>
</div>

<!-- TAB MATMUL -->
<div class="panel" id="panel-matmul">
<h2>Multiplication matricielle GPU</h2>
<div class="grid">
<div class="field"><label>GPU</label><select id="mm-arch">__GPU_OPTIONS__</select></div>
<div class="field"><label>Taille matrice N</label><input type="number" id="mm-n" value="128"></div>
</div>
<button class="btn" onclick="runMatmul()">CALCULER</button>
<div class="resultat" id="out-matmul">Pret.</div>
</div>

<!-- TAB SPARK -->
<div class="panel" id="panel-spark">
<h2>SPARK - Traitement distribue</h2>
<div class="field"><label>Texte a analyser</label><textarea id="sp-text">Nexus Spark Muse est un systeme de calcul distribue. Nexus est rapide. Spark est distribue. Muse est un moteur de recherche. Nexus Spark Muse forme un trio puissant pour le calcul GPU.</textarea></div>
<button class="btn" onclick="runWordCount()">WORD COUNT</button>
<button class="btn sec" onclick="runSparkMap()">MAP x*2</button>
<button class="btn sec" onclick="runSparkFilter()">FILTER pair</button>
<button class="btn sec" onclick="runSparkReduce()">REDUCE sum</button>
<div class="resultat" id="out-spark">Pret.</div>
</div>

<!-- TAB MUSE -->
<div class="panel" id="panel-muse">
<h2>MUSE - Moteur de recherche matriciel</h2>
<div class="field"><label>Documents (un par ligne)</label><textarea id="muse-docs">Nexus est un systeme d'exploration.
Spark traite les donnees massives.
Muse est un moteur de recherche semantique.
CUDA permet le calcul parallele sur GPU.
H100 est une puce NVIDIA pour l'IA.
Rubin est la prochaine generation de GPU.</textarea></div>
<div class="field"><label>Requete</label><input type="text" id="muse-q" value="calcul GPU distribue"></div>
<div class="field"><label>Top K</label><input type="number" id="muse-k" value="3"></div>
<button class="btn" onclick="runMuse()">RECHERCHER</button>
<div class="resultat" id="out-muse">Pret.</div>
</div>

<!-- TAB SPECS -->
<div class="panel" id="panel-specs">
<h2>Specifications GPU NVIDIA</h2>
<div id="out-specs"></div>
</div>

<!-- TAB BENCH -->
<div class="panel" id="panel-bench">
<h2>Benchmark tous GPU</h2>
<div class="grid">
<div class="field"><label>Modele (B params)</label><input type="number" id="bench-params" value="7"></div>
<div class="field"><label>Tokens</label><input type="number" id="bench-tokens" value="1000"></div>
</div>
<button class="btn" onclick="runBench()">BENCHMARKER</button>
<div class="resultat" id="out-bench">Pret.</div>
</div>

</div>

<div class="footer">NEXUS CUDA SPARK MUSE v__V__ - Aissa Mohammedi (DGK) - NEXUS-OPEN-2.0</div>

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

async function runGpu(){
  const d = {
    arch: document.getElementById('gpu-arch').value,
    params_b: parseFloat(document.getElementById('gpu-params').value),
    tokens: parseInt(document.getElementById('gpu-tokens').value),
    batch: parseInt(document.getElementById('gpu-batch').value),
    precision: document.getElementById('gpu-prec').value,
  };
  const r = await fetch('/api/gpu/inference', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-gpu', j, !j.ok);
}

async function runTrain(){
  const d = {
    arch: document.getElementById('tr-arch').value,
    params_b: parseFloat(document.getElementById('tr-params').value),
    batch: parseInt(document.getElementById('tr-batch').value),
    seq: parseInt(document.getElementById('tr-seq').value),
    precision: document.getElementById('tr-prec').value,
  };
  const r = await fetch('/api/gpu/train', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-train', j, !j.ok);
}

async function runMatmul(){
  const d = {
    arch: document.getElementById('mm-arch').value,
    n: parseInt(document.getElementById('mm-n').value),
  };
  const r = await fetch('/api/gpu/matmul', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-matmul', j, !j.ok);
}

async function runWordCount(){
  const d = {text: document.getElementById('sp-text').value};
  const r = await fetch('/api/spark/wordcount', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-spark', j, !j.ok);
}

async function runSparkMap(){
  const r = await fetch('/api/spark/map', {method:'POST'});
  const j = await r.json();
  show('out-spark', j, !j.ok);
}

async function runSparkFilter(){
  const r = await fetch('/api/spark/filter', {method:'POST'});
  const j = await r.json();
  show('out-spark', j, !j.ok);
}

async function runSparkReduce(){
  const r = await fetch('/api/spark/reduce', {method:'POST'});
  const j = await r.json();
  show('out-spark', j, !j.ok);
}

async function runMuse(){
  const docs = document.getElementById('muse-docs').value.split('\n').filter(x => x.trim());
  const d = {
    docs: docs,
    q: document.getElementById('muse-q').value,
    k: parseInt(document.getElementById('muse-k').value),
  };
  const r = await fetch('/api/muse/recherche', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  show('out-muse', j, !j.ok);
}

async function runBench(){
  const d = {
    params_b: parseFloat(document.getElementById('bench-params').value),
    tokens: parseInt(document.getElementById('bench-tokens').value),
  };
  const r = await fetch('/api/bench', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j = await r.json();
  if(j.resultats){
    let html = '<table><tr><th>#</th><th>GPU</th><th>Tokens/s</th><th>t/s/W</th><th>t/s/$</th></tr>';
    j.resultats.forEach((r, i) => {
      html += '<tr><td>' + (i+1) + '</td><td>' + r.gpu + '</td><td>' + r.tokens_par_seconde + '</td><td>' + r.tokens_par_seconde_par_watt + '</td><td>' + r.tokens_par_seconde_par_dollar + '</td></tr>';
    });
    html += '</table>';
    document.getElementById('out-bench').innerHTML = html;
    document.getElementById('out-bench').className = 'resultat';
  } else {
    show('out-bench', j, !j.ok);
  }
}

async function chargerSpecs(){
  const r = await fetch('/api/gpu/specs');
  const j = await r.json();
  let html = '';
  Object.entries(j.gpus).forEach(([k, g]) => {
    html += '<div class="card"><h3>' + g.nom + ' (' + g.architecture + ', ' + g.annee + ')</h3>';
    html += '<div class="stat-row"><span class="stat-label">Transistors</span><span class="stat-value">' + g.transistors_md.toLocaleString() + ' M</span></div>';
    html += '<div class="stat-row"><span class="stat-label">CUDA Cores</span><span class="stat-value">' + g.cuda_cores.toLocaleString() + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Tensor Cores</span><span class="stat-value">' + g.tensor_cores.toLocaleString() + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Memoire</span><span class="stat-value">' + g.memoire_go + ' Go ' + g.memoire_type + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Bande passante</span><span class="stat-value">' + g.bande_passante_gbs + ' GB/s</span></div>';
    html += '<div class="stat-row"><span class="stat-label">FP16</span><span class="stat-value">' + g.tflops_fp16 + ' TFLOPS</span></div>';
    html += '<div class="stat-row"><span class="stat-label">FP8</span><span class="stat-value">' + g.tflops_fp8 + ' TFLOPS</span></div>';
    html += '<div class="stat-row"><span class="stat-label">TDP</span><span class="stat-value">' + g.tdp_w + ' W</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Prix</span><span class="stat-value">$' + g.prix_usd.toLocaleString() + '</span></div>';
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
        if p == "/api/status":
            device = get_device()
            spark = get_spark()
            muse = get_muse()
            return self._json({
                "ok": True,
                "version": VERSION,
                "device": device.resume(),
                "spark": spark.resume(),
                "muse": muse.resume(),
                "historique": ETAT["historique"][-20:],
            })
        if p == "/api/health":
            return self._json({"ok": True, "version": VERSION})
        return self._json({"erreur": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        data = self._read_json()

        if p == "/api/gpu/inference":
            arch = data.get("arch", "h100")
            params_b = float(data.get("params_b", 7))
            tokens = int(data.get("tokens", 1000))
            batch = int(data.get("batch", 1))
            precision = data.get("precision", "fp16")
            device = get_device(arch)
            resultat = device.inference_llm(params_b, tokens, precision, batch)
            ajouter_historique("inference", data, resultat)
            return self._json(resultat)

        if p == "/api/gpu/train":
            arch = data.get("arch", "h100")
            params_b = float(data.get("params_b", 7))
            batch = int(data.get("batch", 8))
            seq = int(data.get("seq", 4096))
            precision = data.get("precision", "fp16")
            device = get_device(arch)
            resultat = device.entrainement_step(params_b, batch, seq, precision)
            ajouter_historique("train", data, resultat)
            return self._json(resultat)

        if p == "/api/gpu/matmul":
            arch = data.get("arch", "h100")
            n = int(data.get("n", 128))
            n = min(n, 256)
            device = get_device(arch)
            import random
            A = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
            B = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
            resultat = device.matmul(A, B)
            ajouter_historique("matmul", {"arch": arch, "n": n}, resultat)
            return self._json(resultat)

        if p == "/api/spark/wordcount":
            text = data.get("text", "")
            spark = get_spark()
            resultat = spark.wordCount(text)
            ajouter_historique("wordcount", {"taille": len(text)}, resultat)
            return self._json(resultat)

        if p == "/api/spark/map":
            spark = get_spark()
            rdd = spark.parallelize(list(range(10)))
            rdd2 = spark.map(rdd, "x*2", lambda x: x * 2)
            return self._json({"ok": True, "rdd_source": rdd["data"], "rdd_map": rdd2["data"], "temps_ms": rdd2.get("temps_ms")})

        if p == "/api/spark/filter":
            spark = get_spark()
            rdd = spark.parallelize(list(range(20)))
            rdd2 = spark.filter(rdd, "pair", lambda x: x % 2 == 0)
            return self._json({"ok": True, "rdd_source": rdd["data"], "rdd_filter": rdd2["data"], "temps_ms": rdd2.get("temps_ms")})

        if p == "/api/spark/reduce":
            spark = get_spark()
            rdd = spark.parallelize(list(range(100)))
            resultat = spark.reduce(rdd, lambda a, b: a + b, 0)
            return self._json(resultat)

        if p == "/api/muse/recherche":
            docs = data.get("docs", [])
            q = data.get("q", "")
            k = int(data.get("k", 3))
            muse = get_muse()
            muse.vecteurs = []
            muse.metadonnees = []
            for i, d in enumerate(docs):
                muse.ajouter(d, {"index": i})
            resultat = muse.rechercher(q, k)
            ajouter_historique("muse", {"q": q, "docs": len(docs)}, resultat)
            return self._json(resultat)

        if p == "/api/bench":
            params_b = float(data.get("params_b", 7))
            tokens = int(data.get("tokens", 1000))
            resultats = []
            for arch_id in GPUS.keys():
                try:
                    device = CudaDevice(arch_id)
                    r = device.inference_llm(params_b, tokens, "fp16", 1)
                    if r.get("ok"):
                        r["arch_id"] = arch_id
                        resultats.append(r)
                except Exception:
                    pass
            resultats.sort(key=lambda x: -x.get("tokens_par_seconde", 0))
            return self._json({"ok": True, "params_b": params_b, "tokens": tokens, "resultats": resultats})

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
    print("  NEXUS CUDA SPARK MUSE v" + VERSION)
    print("  Auteur : " + AUTEUR + "  |  Licence : " + LICENCE)
    print("=" * 78)
    print("")
    print("  Simulateur GPU + Spark + MUSE")
    print("  Reimplementation Python des primitives CUDA.")
    print("")
    print("  GPU disponibles : " + str(len(GPUS)))
    for k, g in GPUS.items():
        print("    " + k + " : " + g["nom"] + " (" + str(g["tflops_fp16"]) + " TFLOPS)")
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