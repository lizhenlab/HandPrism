#!/usr/bin/env python3
"""Compatibility entry; use scripts/train.py for new commands."""

import sys

from scripts import train as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

# Preserve module-level imports and monkeypatching by older local callers.
sys.modules[__name__] = _implementation
