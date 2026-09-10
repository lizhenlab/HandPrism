#!/usr/bin/env python3
"""Compatibility entry; use scripts/build_manifests.py for new commands."""

import sys

from scripts import build_manifests as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

# Preserve module-level imports and monkeypatching by older local callers.
sys.modules[__name__] = _implementation
