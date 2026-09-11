#!/usr/bin/env python3
"""Compatibility entry for the architecture-specific ARCTIC/HOT3D audit."""

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

from scripts.check_readiness import main


if __name__ == "__main__":
    raise SystemExit(main())
