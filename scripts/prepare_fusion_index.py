#!/usr/bin/env python3
"""Build a new frozen ARCTIC/HOT3D difficulty index; never alter original splits.

Only train/val geometry is decoded. Test rows are copied unchanged and are never
used to choose thresholds, sampling weights, augmentations or checkpoints.
"""
from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from handprism.data.arctic import load_arctic_window
from handprism.data.hot3d import load_hot3d_window
from handprism.data.dataset import read_jsonl
from handprism.data.difficulty import (window_strata, select_validation_windows,
    FUSION_INDEX_VERSION, VALIDATION_WINDOW_POLICY)
from handprism.data.policy import allowed_path
from handprism.data.schema import MANIFEST_SCHEMA
from scripts.train import manifest_report, sha256
from scripts.build_manifests import hot3d_valid_ranges, HOT3D_REQUIRED_MASKS


def starts_for_record(record: dict, frames: int, stride: int) -> list[int]:
    if frames < 1 or stride < 1:
        raise ValueError("frames and stride must be positive")
    ranges = record.get("valid_ranges", [[
        record.get("start_min", record.get("start_frame", 0)),
        int(record.get("start_max", int(record["num_frames"]) - frames)) + frames,
    ]])
    starts = []
    for lower, stop in ranges:
        last = int(stop) - frames
        if last < int(lower):
            raise ValueError("invalid complete-window interval in source manifest")
        starts.extend(range(int(lower), last + 1, stride))
        starts.append(last)
    return sorted(set(starts))


def geometry_sample(record: dict, start: int, mano: Path):
    common = dict(mano_model_path=mano, extended_contract=True, decode_rgb=False)
    if record["dataset"] == "arctic":
        return load_arctic_window(record["root"], record["sequence"], start, 81,
                                  image_offset=record["image_offset"], **common)
    if record["dataset"] == "hot3d":
        return load_hot3d_window(record["recording_root"], start, 81,
                                 required_masks=tuple(record["required_masks"]), **common)
    raise ValueError("only ARCTIC and HOT3D are permitted")


def validation_ranges(record: dict) -> list[list[int]]:
    """Certify only the already-held-out recording; never inspect test data.

    Older HOT3D val rows kept one start but omitted ranges, so recover ranges
    from the exact same required masks rather than guessing from num_frames.
    """
    if record["split"] != "val":
        raise ValueError("validation range recovery only accepts val records")
    if record["dataset"] == "arctic":
        ranges = [[int(record["usable_start"]), int(record["usable_stop"])]]
    elif record["dataset"] == "hot3d":
        if set(record["required_masks"]) != set(HOT3D_REQUIRED_MASKS):
            raise ValueError("HOT3D validation mask contract changed")
        count, ranges, stats = hot3d_valid_ranges(allowed_path(record["recording_root"]))
        if count != record["num_frames"] or stats != record["mask_stats"]:
            raise ValueError("HOT3D masks differ from the frozen source manifest")
        ranges = [list(pair) for pair in ranges]
    else:
        raise ValueError("only ARCTIC and HOT3D are permitted")
    previous = -1
    for lo, hi in ranges:
        if lo < 0 or lo < previous or hi - lo < 81 or hi > record["num_frames"]:
            raise ValueError("invalid certified validation range")
        previous = hi
    if not any(lo <= record["start_frame"] and record["start_frame"] + 81 <= hi for lo, hi in ranges):
        raise ValueError("source validation window outside certified ranges")
    return ranges


def expand_validation_record(record: dict, mano: Path, maximum: int = 12) -> list[dict]:
    ranges = validation_ranges(record)
    candidates = []
    for lo, hi in ranges:
        # No tail overlap or duplicate windows. Fixed 81-frame spacing is
        # independent of train difficulty stride and prediction quality.
        for start in range(lo, hi - 81 + 1, 81):
            candidates.append({**record, "start_frame": start, "valid_ranges": ranges,
                "source_validation_start_frame": record["start_frame"],
                "validation_strata": window_strata(geometry_sample(record, start, mano))})
    return select_validation_windows(candidates, maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=81)
    parser.add_argument("--val-windows-per-recording", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source, output = allowed_path(args.source), allowed_path(args.output)
    if args.stride < 1 or args.val_windows_per_recording < 2 or output.exists() or output == source:
        raise ValueError("require positive stride and a new output directory")
    audit = manifest_report(source, ("arctic", "hot3d"))
    report = copy.deepcopy(audit["split_report"])
    if "fusion_index" in report:
        raise ValueError("build from the original/rebased two-source split, not an existing Fusion index")
    report["version"] = MANIFEST_SCHEMA
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".fusion-index-", dir=output.parent))
    validation = {}
    for dataset in ("arctic", "hot3d"):
        for split in ("train", "val", "test"):
            path = source / f"{dataset}_{split}.jsonl"
            rows = read_jsonl(path)
            if split == "test":
                data = path.read_bytes()  # exact byte copy, no test geometry
            else:
                expanded = []
                for index, record in enumerate(rows):
                    if split == "train":
                        windows = []
                        for start in starts_for_record(record, 81, args.stride):
                            tags = window_strata(geometry_sample(record, start, args.mano_model))
                            if tags:
                                windows.append({"start_frame": start, "strata": tags})
                        record["difficulty_windows"] = windows
                    else:
                        expanded.extend(expand_validation_record(record, args.mano_model, args.val_windows_per_recording))
                    print(json.dumps({"dataset": dataset, "split": split, "records": index+1, "total": len(rows)}), flush=True)
                if split == "val":
                    rows = expanded
                    validation[dataset] = {"windows": len(rows),
                        "recordings": len({row["recording_id"] for row in rows}),
                        "strata_counts": {tag: sum(tag in row["validation_strata"] for row in rows)
                                          for tag in ("oos", "edge", "small", "fast", "occluded")}}
                data = ("\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows)+"\n").encode()
            (stage / path.name).write_bytes(data)
            report["datasets"][dataset][split]["sha256"] = hashlib.sha256(data).hexdigest()
            report["datasets"][dataset][split]["rows"] = len(rows)
    report["fusion_index"] = {
        "version": FUSION_INDEX_VERSION, "stride": args.stride, "source_split_report_sha256": audit["split_report_sha256"],
        "source_manifests": audit["manifests"], "thresholds": {"edge_px": 32, "small_diagonal_px": 48, "fast_wrist_m_s": 1.},
        "validation_policy": VALIDATION_WINDOW_POLICY, "validation": validation,
        "val_windows_per_recording": args.val_windows_per_recording,
        "test_used_for_design": False, "timestamp_sources": ["arctic_release_index_30hz", "hot3d_rgb_time_code_ns"],
    }
    (stage / "split_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    manifest_report(stage, ("arctic", "hot3d"))
    if manifest_report(source, ("arctic", "hot3d")) != audit:
        raise RuntimeError("source manifest changed during index construction; refusing to publish")
    stage.rename(output)
    print(json.dumps({"output": str(output), "sha256": sha256(output / "split_report.json")}))


if __name__ == "__main__":
    main()
