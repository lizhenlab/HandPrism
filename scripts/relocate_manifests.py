#!/usr/bin/env python3
"""Relocate frozen ARCTIC/HOT3D paths without resampling or copying datasets."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path

from handprism.data.dataset import read_jsonl
from handprism.data.policy import SUPPORTED_DATASETS, allowed_path, validate_record
from handprism.data.schema import MANIFEST_SCHEMA
from scripts.check_readiness import audit_manifests, required_paths
from scripts.train import manifest_report


def relocated_record(record: dict, roots: dict[str, Path]) -> dict:
    dataset = record["dataset"]
    validate_record(record, Path(f"{dataset}_{record['split']}.jsonl"))
    result = copy.deepcopy(record)
    if dataset == "arctic":
        if record["sequence"] != record["recording_id"] or "recording_root" in record:
            raise ValueError("ARCTIC recording identity or root fields disagree")
        result["root"] = str(roots[dataset])
    else:
        identity = str(record["recording_id"])
        if (not identity or identity in (".", "..") or Path(identity).name != identity
                or "\\" in identity or "root" in record
                or Path(record["recording_root"]).name != identity):
            raise ValueError("HOT3D recording identity must match its directory name")
        result["recording_root"] = str(roots[dataset] / identity)
    validate_record(result, Path(f"{dataset}_{record['split']}.jsonl"))
    return result


def relocate(source: Path, output: Path, arctic_root: Path, hot3d_root: Path) -> dict:
    source = allowed_path(source).resolve()
    output = allowed_path(output).absolute()
    if output.exists() or output.is_symlink() or output == source:
        raise ValueError("output must be a new directory; frozen manifests are never overwritten")
    roots = {name: allowed_path(path).resolve() for name, path in
             zip(SUPPORTED_DATASETS, (arctic_root, hot3d_root))}
    if not all(root.is_dir() for root in roots.values()):
        raise ValueError("dataset roots must be existing extracted directories")
    if not all((roots["arctic"] / "data" / part).is_dir()
               for part in ("meta", "raw_seqs", "images")):
        raise ValueError("ARCTIC root must contain data/{meta,raw_seqs,images}")

    audited = manifest_report(source, SUPPORTED_DATASETS)
    checks = {}
    audit_manifests(source, SUPPORTED_DATASETS, checks, check_data_files=False)
    failed = [key for key, value in checks.items() if not value["pass"]]
    if failed:
        raise ValueError(f"source manifest contract failed: {failed}")

    payloads = {}
    paths = set()
    for dataset in SUPPORTED_DATASETS:
        for split in ("train", "val", "test"):
            name = f"{dataset}_{split}.jsonl"
            rows = [relocated_record(row, roots) for row in read_jsonl(source / name)]
            for row in rows:
                paths.update(required_paths(row))
            payloads[name] = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False) + "\n" for row in rows).encode()

    def exists(path: Path) -> tuple[str, bool]:
        return str(path), allowed_path(path).exists()

    missing = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, (path, present) in enumerate(executor.map(exists, sorted(paths)), 1):
            if not present:
                missing.append(path)
            if index % 250 == 0:
                print(json.dumps({"checked_paths": index, "total_paths": len(paths)}), flush=True)
    if missing:
        raise FileNotFoundError(f"{len(missing)} required paths missing; first 10: {missing[:10]}")

    current = manifest_report(source, SUPPORTED_DATASETS)
    if (current["split_report_sha256"] != audited["split_report_sha256"]
            or current["manifests"] != audited["manifests"]):
        raise RuntimeError("source manifests changed during relocation")
    report = copy.deepcopy(audited["split_report"])
    report["version"] = MANIFEST_SCHEMA
    for name, payload in payloads.items():
        dataset, split = Path(name).stem.rsplit("_", 1)
        report["datasets"][dataset][split]["sha256"] = hashlib.sha256(payload).hexdigest()
    report["relocation"] = {
        "operation": "dataset-root-only",
        "source_split_report_sha256": audited["split_report_sha256"],
        "source_schema_sha256": audited["source_schema_sha256"],
        "source_manifests": audited["manifests"],
        "dataset_roots": {name: str(path) for name, path in roots.items()},
        "required_paths_checked": len(paths),
    }

    # Reserve exclusively. Publish the report last, so interrupted output fails
    # the normal readiness checks. A partial output is retained for inspection.
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(payload)
    with (output / "split_report.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    manifest_report(output, SUPPORTED_DATASETS)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arctic-root", required=True, type=Path)
    parser.add_argument("--hot3d-root", required=True, type=Path)
    args = parser.parse_args()
    report = relocate(args.source, args.output, args.arctic_root, args.hot3d_root)
    print(json.dumps({"output": str(args.output), "relocation": report["relocation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
