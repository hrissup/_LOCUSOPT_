#!/usr/bin/env python3
"""
main.py — LocusOpt P2 entry point (Phase 2, 40% version).

Run with:
    python main.py analyze <source.c> [--func FUNC]

Commands:
    analyze  — Analyse cache locality of a kernel source file
"""

import sys
from optimizer.cli import main

if __name__ == "__main__":
    sys.exit(main())
