"""Public namespace, serialized tensor keys and metadata format contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import handprism
from handprism.config import HandPrismConfig
from handprism.data.contract import HandPrismSample
from handprism.data.dataset import HandPrismWindowDataset
from handprism.data.schema import MANIFEST_SCHEMA, supports_manifest_schema
from handprism.decoder import HandPrismDecoder, HandPrismDecoderOutput
from handprism.losses import HandPrismLoss, HandPrismPrediction, HandPrismTarget
from handprism.model import HandPrismModel, HandPrismOutput
from handprism.system import HandPrismSystem
from scripts.train import manifest_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint", [
    "train.py", "evaluate.py", "infer.py", "run_pipeline.py", "check_readiness.py",
    "audit_readiness.py", "build_manifests.py", "prepare_fusion_index.py",
    "export_core_weights.py", "make_v3_ablation.py", "validate_v3_cpu.py",
    "validate_fusion_runtime.py",
    "relocate_manifests.py",
])
def test_cli_imports_without_pythonpath_from_another_directory(entrypoint, tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / entrypoint), "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_one_public_package_and_consistent_version():
    packages = {path.name for path in (ROOT/"src").iterdir() if path.is_dir() and (path/"__init__.py").exists()}
    assert packages == {"handprism"}
    version = re.search(r'^version = "([^"]+)"$', (ROOT/"pyproject.toml").read_text(), re.MULTILINE)
    assert version and handprism.__version__ == version.group(1)
    assert Path(handprism.__file__).resolve() == ROOT/"src/handprism/__init__.py"


def test_project_classes_have_canonical_names():
    for kind in (HandPrismConfig, HandPrismSample, HandPrismWindowDataset,
                 HandPrismDecoder, HandPrismDecoderOutput, HandPrismLoss,
                 HandPrismPrediction, HandPrismTarget, HandPrismModel,
                 HandPrismOutput, HandPrismSystem):
        assert kind.__name__.startswith("HandPrism")
        assert kind.__module__.startswith("handprism.")


@pytest.mark.parametrize("value", [None, 123, "unregistered-dataset-mixture-v2-clean"])
def test_unknown_manifest_schema_is_rejected(value):
    assert supports_manifest_schema(MANIFEST_SCHEMA)
    assert not supports_manifest_schema(value)


def test_manifest_metadata_identity_does_not_relax_checksums(tmp_path):
    report = {"version": MANIFEST_SCHEMA, "selected_datasets": ["arctic", "hot3d"], "datasets": {}}
    for dataset in ("arctic", "hot3d"):
        report["datasets"][dataset] = {}
        for split in ("train", "val", "test"):
            row = dict(dataset=dataset, split=split, recording_id=f"{dataset}-{split}", split_group=split)
            payload = (json.dumps(row) + "\n").encode()
            (tmp_path/f"{dataset}_{split}.jsonl").write_bytes(payload)
            report["datasets"][dataset][split] = dict(rows=1, recordings=1, split_groups=1,
                sha256=hashlib.sha256(payload).hexdigest())
    path = tmp_path/"split_report.json"
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    audited = manifest_report(tmp_path, ("arctic", "hot3d"))
    assert audited["split_report"]["version"] == MANIFEST_SCHEMA
    assert audited["source_schema_sha256"] == hashlib.sha256(MANIFEST_SCHEMA.encode()).hexdigest()
    assert path.read_bytes() == original
    report["datasets"]["arctic"]["test"]["sha256"] = "0" * 64
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="manifest does not match"):
        manifest_report(tmp_path, ("arctic", "hot3d"))
