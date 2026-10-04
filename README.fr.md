# NEXUS CUDA KILLER

> Portable Python simulation of GPU optimization logic and high-performance compute patterns.

[![License](https://img.shields.io/badge/license-NEXUS--OPEN--2.0-00ff88)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-00d4ff)](https://python.org)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-ff00c8)](.)  
[![Version](https://img.shields.io/badge/version-1.0.0-c8a45c)](.)  

Author: Aissa Mohammedi (DGK)  
License: NEXUS-OPEN-2.0  
Version: 1.0.0

---

## Project vision

This repository studies GPU optimization logic in portable Python.

GPU performance patterns are not magic:
- reducing memory passes
- keeping running max/sum values
- batching independent work
- exploiting vectorized compute
- validating integrity with compact structures
- minimizing redundant calculations

---

## Included versions

- `main`: clean core version (online softmax + hash + benchmark)
- `versions/v1`: CUDA Spark Muse (7 GPU architectures + Spark + vector search)
- `versions/v2`: instability calculator (real CUDA test + multi-OS analysis)
- `versions/v3`: future sandbox

---

## Quick start

```bash
git clone https://github.com/aipromotion20102017-create/nexus-cuda-killer.git
cd nexus-cuda-killer
python3 nexus_cuda_killer.py
```

Open: `http://localhost:8103/`

---

## Features

- Online softmax (naive vs optimized)
- SHA-256 batch hashing + Merkle trees
- Matrix multiplication simulation
- LLM inference simulation
- GPU/CPU benchmarking
- Web UI + REST API

---

## License

NEXUS-OPEN-2.0 - See LICENSE

---

## Author

Aissa Mohammedi (DGK)
