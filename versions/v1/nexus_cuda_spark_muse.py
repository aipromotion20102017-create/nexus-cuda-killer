#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# NEXUS CUDA INSTABILITY CALCULATOR v1.0.0
# Test CUDA reel + calcul d'instabilite pour fusion multi-OS.
# Auteur : Aissa Mohammedi (DGK)
# Licence : NEXUS-OPEN-2.0
# Compatible a-Shell iOS - os.path uniquement - stdlib pure (torch/cupy optionnels)
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import json
import math
import socket
import hashlib
import platform
import datetime
import threading
import subprocess
import collections
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any

VERSION = "1.0.0"
AUTEUR = "Aissa Mohammedi (DGK)"
LICENCE = "NEXUS-OPEN-2.0"

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "Documents", "nexus_cuda_instability")
os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, "rapports"), exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    try:
        print(line, flush=True)
    except Exception:
        pass


def sep(titre=""):
    print("")
    print("=" * 66)
    if titre:
        print("  " + titre)
        print("=" * 66)


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 1 : DEPENDANCES OPTIONNELLES
# ═══════════════════════════════════════════════════════════════════════════

HAS_PSUTIL = False
HAS_TORCH = False
HAS_CUPY = False
HAS_NUMBA = False
HAS_NVIDIA_SMI = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    pass

try:
    from numba import cuda as numba_cuda
    HAS_NUMBA = True
except ImportError:
    pass


def detecter_nvidia_smi():
    """Detecte nvidia-smi sur le systeme."""
    global HAS_NVIDIA_SMI
    for chemin in ["nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"]:
        try:
            r = subprocess.run([chemin, "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                HAS_NVIDIA_SMI = True
                return r.stdout.strip()
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 2 : DETECTION MATERIEL REELLE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Materiel:
    os: str = ""
    os_version: str = ""
    os_detail: str = ""
    machine: str = ""
    architecture: str = ""
    python_version: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_freq_mhz: float = 0.0
    ram_total_go: float = 0.0
    ram_dispo_go: float = 0.0
    disk_total_go: float = 0.0
    disk_free_go: float = 0.0
    gpu_detecte: str = ""
    gpu_driver: str = ""


def detecter_materiel() -> Materiel:
    sep("DETECTION MATERIEL REELLE")

    m = Materiel()
    m.os = platform.system()
    m.os_version = platform.release()
    m.machine = platform.machine()
    m.architecture = platform.processor()
    m.python_version = platform.python_version()

    # OS detail
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    m.os_detail = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        m.os_detail = m.os + " " + m.os_version

    # CPU info
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            contenu = f.read()
            match = re.search(r"model name\s*:\s*(.+)", contenu)
            if match:
                m.cpu_model = match.group(1).strip()
            freq = re.search(r"cpu MHz\s*:\s*([\d.]+)", contenu)
            if freq:
                m.cpu_freq_mhz = round(float(freq.group(1)), 1)
    except (OSError, NameError):
        m.cpu_model = platform.processor()

    # Cores / RAM / Disk
    if HAS_PSUTIL:
        m.cpu_cores = psutil.cpu_count(logical=False) or 0
        m.cpu_threads = psutil.cpu_count(logical=True) or 0
        mem = psutil.virtual_memory()
        m.ram_total_go = round(mem.total / (1024**3), 2)
        m.ram_dispo_go = round(mem.available / (1024**3), 2)
        disk = psutil.disk_usage("/")
        m.disk_total_go = round(disk.total / (1024**3), 2)
        m.disk_free_go = round(disk.free / (1024**3), 2)
    else:
        m.cpu_cores = os.cpu_count() or 0
        m.cpu_threads = m.cpu_cores

    # GPU
    gpu_info = detecter_nvidia_smi()
    if gpu_info:
        parts = gpu_info.split(",")
        if len(parts) >= 3:
            m.gpu_detecte = parts[0].strip()
            m.gpu_driver = parts[2].strip()

    print("  OS            : " + m.os + " " + m.os_version)
    print("  Detail        : " + m.os_detail)
    print("  Machine       : " + m.machine)
    print("  Python        : " + m.python_version)
    print("  CPU           : " + m.cpu_model)
    print("  Cores         : " + str(m.cpu_cores) + " / " + str(m.cpu_threads) + " threads")
    if m.cpu_freq_mhz:
        print("  Frequence     : " + str(m.cpu_freq_mhz) + " MHz")
    print("  RAM           : " + str(m.ram_total_go) + " Go (" + str(m.ram_dispo_go) + " Go dispo)")
    print("  Disque        : " + str(m.disk_total_go) + " Go (" + str(m.disk_free_go) + " Go libre)")
    if m.gpu_detecte:
        print("  GPU           : " + m.gpu_detecte + " (driver " + m.gpu_driver + ")")
    else:
        print("  GPU           : aucun NVIDIA detecte")

    return m


import re  # import apres usage dans detecter_materiel


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 3 : TEST CUDA REEL - PYTORCH
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestCUDA:
    source: str = ""
    dispo: bool = False
    gpu_nom: str = ""
    capability: Tuple[int, int] = (0, 0)
    vram_go: float = 0.0
    matmul_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gops: float = 0.0
    erreurs_stabilite: int = 0
    stabilite_pourcent: float = 0.0
    erreur: str = ""


def test_cuda_pytorch() -> TestCUDA:
    sep("TEST CUDA REEL - PyTorch")

    t = TestCUDA(source="pytorch")

    if not HAS_TORCH:
        t.erreur = "PyTorch non installe"
        print("  [KO] PyTorch non installe.")
        return t

    t.dispo = torch.cuda.is_available()

    if not t.dispo:
        t.erreur = "CUDA non disponible"
        print("  [KO] CUDA non disponible.")
        return t

    t.gpu_nom = torch.cuda.get_device_name(0)
    t.capability = torch.cuda.get_device_capability(0)
    t.vram_go = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    print("  [OK] CUDA disponible")
    print("  GPU : " + t.gpu_nom)
    print("  Compute Capability : " + str(t.capability[0]) + "." + str(t.capability[1]))
    print("  VRAM : " + str(t.vram_go) + " Go")

    # Test matmul 4096x4096
    try:
        N = 4096
        A = torch.randn(N, N, device="cuda")
        B = torch.randn(N, N, device="cuda")

        for _ in range(3):
            C = A @ B
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(10):
            C = A @ B
        torch.cuda.synchronize()
        elapsed = time.time() - start

        t.matmul_ms = round(elapsed / 10 * 1000, 2)
        t.tflops = round((2 * N**3) / (t.matmul_ms * 1e-3) / 1e12, 2)

        print("")
        print("  Matmul " + str(N) + "x" + str(N))
        print("    Temps : " + str(t.matmul_ms) + " ms")
        print("    Perf  : " + str(t.tflops) + " TFLOPS")
    except Exception as e:
        t.erreur = str(e)[:200]

    # Bande passante memoire
    try:
        x = torch.randn(1024 * 1024 * 256, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        y = x * 2
        torch.cuda.synchronize()
        elapsed = time.time() - start
        bw = (x.element_size() * x.nelement() * 2) / elapsed / 1e9
        t.bandwidth_gops = round(bw, 1)
        print("")
        print("  Bande passante memoire : " + str(t.bandwidth_gops) + " Go/s")
    except Exception as e:
        t.erreur = str(e)[:200]

    # Test stabilite 10 iterations
    erreurs = 0
    for i in range(10):
        try:
            _ = torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")
            torch.cuda.synchronize()
        except Exception as e:
            erreurs += 1
            print("  [KO] Erreur iteration " + str(i) + " : " + str(e)[:80])

    t.erreurs_stabilite = erreurs
    t.stabilite_pourcent = round((1 - erreurs / 10) * 100, 1)
    print("")
    print("  Stabilite GPU : " + str(t.stabilite_pourcent) + " %")

    return t


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 4 : TEST CUDA REEL - CuPy
# ═══════════════════════════════════════════════════════════════════════════

def test_cuda_cupy() -> TestCUDA:
    sep("TEST CUDA REEL - CuPy")

    t = TestCUDA(source="cupy")

    if not HAS_CUPY:
        t.erreur = "CuPy non installe (optionnel)"
        print("  [i] CuPy non installe (optionnel).")
        return t

    try:
        dev = cp.cuda.Device(0)
        t.dispo = True
        t.vram_go = round(dev.mem_info[1] / (1024**3), 2)

        print("  [OK] CuPy actif")
        print("  VRAM : " + str(t.vram_go) + " Go")

        N = 4096
        A = cp.random.randn(N, N, dtype=cp.float32)
        B = cp.random.randn(N, N, dtype=cp.float32)

        for _ in range(3):
            C = cp.dot(A, B)
        cp.cuda.Stream.null.synchronize()

        start = time.time()
        for _ in range(10):
            C = cp.dot(A, B)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start

        t.matmul_ms = round(elapsed / 10 * 1000, 2)
        t.tflops = round((2 * N**3) / (t.matmul_ms * 1e-3) / 1e12, 2)

        print("  Matmul " + str(N) + "x" + str(N) + " : " + str(t.matmul_ms) + " ms - " + str(t.tflops) + " TFLOPS")
    except Exception as e:
        t.erreur = str(e)[:200]
        print("  [KO] Erreur CuPy : " + str(e)[:100])

    return t


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 5 : TEST CPU REEL
# ═══════════════════════════════════════════════════════════════════════════

def test_cpu_benchmark() -> Dict[str, Any]:
    sep("BENCHMARK CPU REEL")

    resultats = {"tests": [], "score_global": 0.0}

    # Test 1 : additions
    t0 = time.time()
    total = 0
    for i in range(10000000):
        total += i * i
    dt1 = round((time.time() - t0) * 1000, 2)
    ops_sec1 = round(10000000 / (dt1 / 1000), 2)
    resultats["tests"].append({
        "nom": "additions",
        "ops": 10000000,
        "temps_ms": dt1,
        "ops_par_sec": ops_sec1,
    })
    print("  Additions 10M : " + str(dt1) + " ms (" + str(ops_sec1) + " ops/s)")

    # Test 2 : hash SHA-256
    t0 = time.time()
    for i in range(100000):
        hashlib.sha256(str(i).encode()).hexdigest()
    dt2 = round((time.time() - t0) * 1000, 2)
    ops_sec2 = round(100000 / (dt2 / 1000), 2)
    resultats["tests"].append({
        "nom": "sha256",
        "ops": 100000,
        "temps_ms": dt2,
        "ops_par_sec": ops_sec2,
    })
    print("  SHA-256 100k : " + str(dt2) + " ms (" + str(ops_sec2) + " hashes/s)")

    # Test 3 : JSON
    data = {"test": list(range(1000))}
    t0 = time.time()
    for i in range(10000):
        json.dumps(data)
    dt3 = round((time.time() - t0) * 1000, 2)
    ops_sec3 = round(10000 / (dt3 / 1000), 2)
    resultats["tests"].append({
        "nom": "json_dumps",
        "ops": 10000,
        "temps_ms": dt3,
        "ops_par_sec": ops_sec3,
    })
    print("  JSON 10k : " + str(dt3) + " ms (" + str(ops_sec3) + " dumps/s)")

    resultats["score_global"] = round((ops_sec1 / 1e6 + ops_sec2 / 1e3 + ops_sec3 / 1e2) / 3, 2)
    print("  Score global : " + str(resultats["score_global"]))

    return resultats


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 6 : CALCUL D'INSTABILITE MULTI-OS
# ═══════════════════════════════════════════════════════════════════════════

class InstabilityCalculator:
    """
    Calcule le score d'instabilite pour fusion macOS + Linux Mint + Windows.
    Base sur des contraintes REELLES de systeme d'exploitation.
    """

    def __init__(self, materiel: Materiel, cuda: TestCUDA):
        self.m = materiel
        self.c = cuda
        self.risques = []

    def _noyaux(self) -> Tuple[float, str]:
        """3 noyaux incompatibles : XNU, Linux, NT."""
        return 5.0, "Noyaux incompatibles (XNU, Linux, NT) - aucune fusion possible"

    def _fs(self) -> Tuple[float, str]:
        """APFS, ext4, NTFS - aucun natif commun."""
        return 10.0, "3 systemes de fichiers incompatibles (APFS, ext4, NTFS)"

    def _bootloader(self) -> Tuple[float, str]:
        """GRUB vs BOOTMGR vs EFI Apple - 3 bootloaders en conflit."""
        return 15.0, "Conflit bootloader (GRUB vs BOOTMGR vs EFI Apple)"

    def _ram(self) -> Tuple[float, str]:
        """macOS ~4 Go + Mint ~1.5 Go + Windows ~3.5 Go = ~9 Go minimum."""
        besoin = 9.0
        ram = self.m.ram_total_go
        if ram == 0:
            return 50.0, "RAM inconnue"
        if ram < besoin:
            return 100.0, "RAM insuffisante (" + str(ram) + " Go < " + str(besoin) + " Go)"
        elif ram < besoin * 1.5:
            return 60.0, "RAM juste (" + str(ram) + " Go pour " + str(besoin) + " Go)"
        else:
            return 10.0, "RAM suffisante (" + str(ram) + " Go)"

    def _disque(self) -> Tuple[float, str]:
        """macOS ~30 Go + Mint ~15 Go + Windows ~40 Go = ~85 Go."""
        besoin = 85.0
        free = self.m.disk_free_go
        if free == 0:
            return 50.0, "Disque inconnu"
        if free < besoin:
            return 100.0, "Disque insuffisant (" + str(free) + " Go < " + str(besoin) + " Go)"
        elif free < besoin * 1.5:
            return 40.0, "Disque juste (" + str(free) + " Go)"
        else:
            return 5.0, "Disque suffisant (" + str(free) + " Go)"

    def _gpu_cuda(self) -> Tuple[float, str]:
        """CUDA n'existe pas sur macOS moderne. Windows + Linux OK."""
        if not self.c.dispo:
            return 80.0, "Pas de CUDA detecte - instabilite GPU"
        tflops = self.c.tflops
        if tflops < 5:
            return 50.0, "GPU faible (" + str(tflops) + " TFLOPS)"
        elif tflops < 30:
            return 20.0, "GPU moyen (" + str(tflops) + " TFLOPS)"
        else:
            return 5.0, "GPU puissant (" + str(tflops) + " TFLOPS)"

    def _stab_gpu(self) -> Tuple[float, str]:
        stab = self.c.stabilite_pourcent
        if stab == 0:
            return 50.0, "Stabilite GPU inconnue"
        if stab >= 100:
            return 0.0, "GPU 100% stable"
        elif stab >= 90:
            return 20.0, "GPU stable (" + str(stab) + "%)"
        elif stab >= 70:
            return 50.0, "GPU instable (" + str(stab) + "%)"
        else:
            return 90.0, "GPU tres instable (" + str(stab) + "%)"

    def _securite(self) -> Tuple[float, str]:
        """SIP + Defender + AppArmor - 3 systemes en conflit."""
        return 25.0, "3 systemes de securite en conflit"

    def _drivers(self) -> Tuple[float, str]:
        """kext + WDM + .ko - aucun driver commun."""
        return 40.0, "Drivers incompatibles (kext vs WDM vs .ko)"

    def _gestion_paquets(self) -> Tuple[float, str]:
        """Homebrew + APT + MSI - 3 gestionnaires incompatibles."""
        return 30.0, "3 gestionnaires de paquets (Homebrew, APT, MSI)"

    def _permissions(self) -> Tuple[float, str]:
        """POSIX 755 vs NTFS ACL vs APFS ACL - 3 modeles."""
        return 20.0, "3 modeles de permissions incompatibles"

    def _timeline(self) -> Tuple[float, str]:
        """Historique : personne n'a reussi cette fusion."""
        return 35.0, "Aucun precedent de fusion 3-OS en production"

    def _licence(self) -> Tuple[float, str]:
        """EULA macOS + GPL Mint + EULA Windows - conflits legaux."""
        return 25.0, "Conflits de licences (Apple EULA vs GPL vs Microsoft EULA)"

    def calculer(self) -> Dict[str, Any]:
        sep("CALCUL D'INSTABILITE MULTI-OS")

        tests = [
            ("Noyaux", self._noyaux()),
            ("Systemes de fichiers", self._fs()),
            ("Bootloader", self._bootloader()),
            ("RAM", self._ram()),
            ("Disque", self._disque()),
            ("GPU / CUDA", self._gpu_cuda()),
            ("Stabilite GPU", self._stab_gpu()),
            ("Securite", self._securite()),
            ("Drivers", self._drivers()),
            ("Gestion paquets", self._gestion_paquets()),
            ("Permissions", self._permissions()),
            ("Precedents", self._timeline()),
            ("Licences", self._licence()),
        ]

        scores = []
        print("")
        for nom, (score, msg) in tests:
            print("  " + nom.ljust(22) + " : " + str(score).rjust(5) + "% - " + msg)
            scores.append(score)
            self.risques.append({"nom": nom, "score": score, "message": msg})

        global_score = round(sum(scores) / len(scores), 1)
        verdict = self._verdict(global_score)

        print("")
        print("-" * 66)
        print("  SCORE D'INSTABILITE GLOBAL : " + str(global_score) + "%")
        print("  VERDICT : " + verdict["titre"])
        print("  " + verdict["message"])
        print("-" * 66)

        return {
            "score_global": global_score,
            "verdict": verdict,
            "details": self.risques,
        }

    def _verdict(self, score: float) -> Dict[str, str]:
        if score < 20:
            return {"titre": "STABLE", "message": "Le systeme peut fonctionner."}
        elif score < 40:
            return {"titre": "RISQUE", "message": "Ca peut marcher, avec beaucoup de travail."}
        elif score < 60:
            return {"titre": "INSTABLE", "message": "Le systeme va planter regulierement."}
        elif score < 80:
            return {"titre": "TRES INSTABLE", "message": "Le systeme est inutilisable en production."}
        else:
            return {"titre": "IMPOSSIBLE", "message": "Ce systeme ne peut pas exister."}


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 7 : ANALYSE ALTERNATIVE
# ═══════════════════════════════════════════════════════════════════════════

def suggerer_alternatives(instab: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggere des alternatives realistes."""
    alts = [
        {
            "nom": "Dual-boot 2 OS",
            "description": "Installer 2 OS sur 2 partitions separees. Choisir au demarrage via GRUB.",
            "score_instabilite": "20%",
            "faisable": True,
            "avantage": "Standard, bien documente, stable",
            "inconvenient": "1 seul OS actif a la fois",
        },
        {
            "nom": "Triple-boot 3 OS",
            "description": "Installer 3 OS sur 3 partitions. GRUB gere les 3.",
            "score_instabilite": "35%",
            "faisable": True,
            "avantage": "3 OS disponibles, chacun natif",
            "inconvenient": "Partitionnement complexe, GRUB fragile",
        },
        {
            "nom": "Virtualisation (KVM/VirtualBox)",
            "description": "1 OS hote + 2 OS invites en VM.",
            "score_instabilite": "25%",
            "faisable": True,
            "avantage": "3 OS simultanes, isoles",
            "inconvenient": "Performance reduite (surtout GPU)",
        },
        {
            "nom": "Conteneurs (LXC/Docker)",
            "description": "1 OS hote + environnements isoles.",
            "score_instabilite": "10%",
            "faisable": True,
            "avantage": "Rapide, stable, partage kernel",
            "inconvenient": "Pas de macOS (kernel different)",
        },
        {
            "nom": "Fusion 3-OS en 1 systeme",
            "description": "Faire tourner macOS + Mint + Windows en meme temps sur le meme noyau.",
            "score_instabilite": str(instab.get("score_global", "?")) + "%",
            "faisable": False,
            "avantage": "Aucun precedent",
            "inconvenient": "Noyaux incompatibles - impossible",
        },
    ]
    return alts


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 8 : RAPPORT FINAL
# ═══════════════════════════════════════════════════════════════════════════

def rapport_final(m: Materiel, c: TestCUDA, cpu: Dict[str, Any],
                 instab: Dict[str, Any], alts: List[Dict[str, str]]) -> str:
    sep("RAPPORT FINAL")

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    doc = {
        "version": VERSION,
        "auteur": AUTEUR,
        "licence": LICENCE,
        "timestamp": ts,
        "materiel": asdict(m),
        "cuda": asdict(c),
        "cpu_benchmark": cpu,
        "instabilite": instab,
        "alternatives": alts,
    }

    # Hash SHA-256
    doc["hash_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True, default=str).encode()
    ).hexdigest()

    # Fichiers
    nom_base = "nexus_cuda_rapport_" + str(int(time.time()))
    json_path = os.path.join(BASE, "rapports", nom_base + ".json")
    md_path = os.path.join(BASE, "rapports", nom_base + ".md")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
    except OSError as e:
        log("Erreur JSON : " + str(e))

    # Markdown
    md = []
    md.append("# NEXUS CUDA INSTABILITY CALCULATOR v" + VERSION)
    md.append("")
    md.append("**Auteur :** " + AUTEUR)
    md.append("**Licence :** " + LICENCE)
    md.append("**Date :** " + ts)
    md.append("**SHA-256 :** `" + doc["hash_sha256"] + "`")
    md.append("")
    md.append("## Materiel")
    md.append("")
    md.append("| Champ | Valeur |")
    md.append("|-------|--------|")
    md.append("| OS | " + m.os + " " + m.os_version + " |")
    md.append("| Detail | " + m.os_detail + " |")
    md.append("| Machine | " + m.machine + " |")
    md.append("| Python | " + m.python_version + " |")
    md.append("| CPU | " + m.cpu_model + " |")
    md.append("| Cores | " + str(m.cpu_cores) + " / " + str(m.cpu_threads) + " |")
    md.append("| RAM | " + str(m.ram_total_go) + " Go |")
    md.append("| Disque | " + str(m.disk_total_go) + " Go (" + str(m.disk_free_go) + " Go libre) |")
    if m.gpu_detecte:
        md.append("| GPU | " + m.gpu_detecte + " |")
    md.append("")

    md.append("## Test CUDA")
    md.append("")
    md.append("- **Source :** " + c.source)
    md.append("- **Disponible :** " + ("oui" if c.dispo else "non"))
    if c.dispo:
        md.append("- **GPU :** " + c.gpu_nom)
        md.append("- **VRAM :** " + str(c.vram_go) + " Go")
        md.append("- **Matmul 4096x4096 :** " + str(c.matmul_ms) + " ms")
        md.append("- **TFLOPS :** " + str(c.tflops))
        md.append("- **Bande passante :** " + str(c.bandwidth_gops) + " Go/s")
        md.append("- **Stabilite :** " + str(c.stabilite_pourcent) + " %")
    md.append("")

    md.append("## Benchmark CPU")
    md.append("")
    md.append("| Test | Ops | Temps (ms) | Ops/sec |")
    md.append("|------|-----|------------|---------|")
    for t in cpu.get("tests", []):
        md.append("| " + t["nom"] + " | " + str(t["ops"]) + " | " + str(t["temps_ms"]) + " | " + str(t["ops_par_sec"]) + " |")
    md.append("")

    md.append("## Instabilite Multi-OS")
    md.append("")
    md.append("**Score global :** " + str(instab["score_global"]) + " %")
    md.append("")
    md.append("**Verdict :** " + instab["verdict"]["titre"] + " - " + instab["verdict"]["message"])
    md.append("")
    md.append("| Composant | Score | Message |")
    md.append("|-----------|-------|---------|")
    for r in instab["details"]:
        md.append("| " + r["nom"] + " | " + str(r["score"]) + "% | " + r["message"] + " |")
    md.append("")

    md.append("## Alternatives realistes")
    md.append("")
    for a in alts:
        md.append("### " + a["nom"])
        md.append("")
        md.append("- **Description :** " + a["description"])
        md.append("- **Score instabilite :** " + a["score_instabilite"])
        md.append("- **Faisable :** " + ("oui" if a["faisable"] else "non"))
        md.append("- **Avantage :** " + a["avantage"])
        md.append("- **Inconvenient :** " + a["inconvenient"])
        md.append("")

    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
    except OSError as e:
        log("Erreur MD : " + str(e))

    print("")
    print("  Rapport JSON : " + json_path)
    print("  Rapport MD   : " + md_path)
    print("")
    print("  Resume :")
    print("    OS       : " + m.os + " " + m.os_version)
    print("    CPU      : " + str(m.cpu_cores) + " cores / " + str(m.cpu_threads) + " threads")
    print("    RAM      : " + str(m.ram_total_go) + " Go")
    print("    Disque   : " + str(m.disk_free_go) + " Go libre")
    print("    CUDA     : " + ("oui" if c.dispo else "non"))
    if c.dispo:
        print("    GPU      : " + c.gpu_nom + " (" + str(c.tflops) + " TFLOPS)")
    print("    Instab.  : " + str(instab["score_global"]) + " %")
    print("    Verdict  : " + instab["verdict"]["titre"])

    return json_path


# ═══════════════════════════════════════════════════════════════════════════
# PARTIE 9 : MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("")
    print("+" + "-" * 62 + "+")
    print("|" + " " * 62 + "|")
    print("|  NEXUS CUDA INSTABILITY CALCULATOR v" + VERSION + "               |")
    print("|" + " " * 62 + "|")
    print("|  Test CUDA reel + calcul d'instabilite multi-OS             |")
    print("|  Auteur : " + AUTEUR + "                        |")
    print("|" + " " * 62 + "|")
    print("+" + "-" * 62 + "+")
    print("")

    # 1. Materiel
    m = detecter_materiel()

    # 2. Test CUDA
    c_pytorch = test_cuda_pytorch()
    c_cupy = test_cuda_cupy()

    cuda_final = c_pytorch if c_pytorch.dispo else c_cupy

    # 3. Benchmark CPU
    cpu = test_cpu_benchmark()

    # 4. Instabilite multi-OS
    calc = InstabilityCalculator(m, cuda_final)
    instab = calc.calculer()

    # 5. Alternatives
    alts = suggerer_alternatives(instab)

    sep("ALTERNATIVES REALISTES")
    for a in alts:
        statut = "[FAISABLE]" if a["faisable"] else "[IMPOSSIBLE]"
        print("")
        print("  " + statut + " " + a["nom"])
        print("     " + a["description"])
        print("     Instabilite : " + a["score_instabilite"])
        print("     Avantage : " + a["avantage"])
        print("     Inconvenient : " + a["inconvenient"])

    # 6. Rapport
    rapport_final(m, cuda_final, cpu, instab, alts)

    print("")
    sep("TERMINE")
    print("")
    print("  Le rapport JSON contient tous les resultats bruts.")
    print("  Envoie-le pour analyse approfondie si besoin.")
    print("")


if __name__ == "__main__":
    main()