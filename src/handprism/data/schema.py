"""Canonical metadata names with a strict read-only legacy format allowlist."""

from __future__ import annotations

import hashlib


MANIFEST_SCHEMA = "handprism-dataset-mixture-v2-clean"

# Fingerprints of two previously issued wire-format labels. This is only a
# metadata compatibility check: dataset scope, file hashes, counts and split
# isolation must still pass the full manifest audit. Writers use the canonical
# label exclusively and never modify the original manifests in place.
LEGACY_MANIFEST_SCHEMA_SHA256 = frozenset({
    "c0a28746fe264ca3d8c59a7572b5dfe58ba9c024ffa93859980d3bd3d21089ae",
    "fb82b8c413b72fa6946cf97dd372a26e9c5329098b57e5062cea413290e26de9",
})


def supports_manifest_schema(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value == MANIFEST_SCHEMA or hashlib.sha256(value.encode()).hexdigest() in LEGACY_MANIFEST_SCHEMA_SHA256
