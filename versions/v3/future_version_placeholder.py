#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NEXUS CUDA INSTABILITY CALCULATOR v1.0.0
# Test CUDA + calcul d'instabilite multi-OS
# Auteur : Aissa Mohammedi (DGK)
# Licence : NEXUS-OPEN-2.0

import os
import sys
import time
import json
import platform

VERSION = "1.0.0"
AUTEUR = "Aissa Mohammedi (DGK)"
LICENCE = "NEXUS-OPEN-2.0"

print("="*72)
print(f"NEXUS CUDA INSTABILITY CALCULATOR v{VERSION}")
print(f"Auteur: {AUTEUR}")
print(f"Licence: {LICENCE}")
print("="*72)
print()
print("Detecteur de materiel et analyseur d'instabilite multi-OS")
print()
print(f"OS: {platform.system()}")
print(f"Architecture: {platform.processor()}")
print(f"Python: {platform.python_version()}")
print()
print("Analyseur d'instabilite pour fusion macOS + Linux Mint + Windows")
print()
