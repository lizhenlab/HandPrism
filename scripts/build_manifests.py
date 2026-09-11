#!/usr/bin/env python3
"""Build deterministic, capability-safe ARCTIC/HOT3D manifests."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import use_workspace
    use_workspace()

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


FRAMES = 81
SEED = 260820308
ARCTIC_BOUNDARY_MARGIN = 10
HOT3D_STREAM = "214-1"
HOT3D_REQUIRED_MASKS = (
    "mask_hand_pose_available",
    "mask_headset_pose_available",
    "mask_good_exposure",
    "mask_qa_pass",
)
FULL_CAPABILITIES = (
    "mano",
    "joints_root_3d",
    "camera_3d",
    "exact_2d",
    "ray",
    "existence",
    "visibility",
)
from handprism.data.policy import SUPPORTED_DATASETS, allowed_path
from handprism.data.schema import MANIFEST_SCHEMA


def score(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).hexdigest()


def fixed_start(identity: str, lower: int, upper: int) -> int:
    if upper < lower:
        raise ValueError(f"{identity} has no {FRAMES}-frame window in [{lower},{upper}]")
    return lower + int(score("start", identity), 16) % (upper - lower + 1)


def fixed_start_in_ranges(identity: str, ranges: Sequence[tuple[int, int]]) -> int:
    """Select uniformly from all valid starts in half-open frame ranges."""

    counts = [stop - start - FRAMES + 1 for start, stop in ranges]
    total = sum(max(0, value) for value in counts)
    if total <= 0:
        raise ValueError(f"{identity} has no valid {FRAMES}-frame range")
    selected = int(score("start", identity), 16) % total
    for (start, _), count in zip(ranges, counts):
        if selected < count:
            return start + selected
        selected -= count
    raise AssertionError("unreachable valid-range selection")


def validation_groups(values: Iterable[str], namespace: str) -> set[str]:
    groups = sorted(set(values), key=lambda value: score(namespace, value))
    if not groups:
        raise ValueError("cannot split an empty group set")
    count = max(1, round(len(groups) * 0.05))
    return set(groups[:count])


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(payload)
            digest.update(payload.encode())
            count += 1
    temporary.replace(path)
    return count, digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _numeric_image_span(path: Path) -> tuple[int, int, int]:
    stems = [
        int(item.stem)
        for item in path.iterdir()
        if item.suffix.lower() in {".jpg", ".jpeg", ".png"} and item.stem.isdigit()
    ]
    if not stems:
        raise FileNotFoundError(f"no numbered RGB images under {path}")
    if len(set(stems)) != len(stems):
        raise RuntimeError(f"duplicate numbered RGB images under {path}")
    if len(stems) != max(stems) - min(stems) + 1:
        raise RuntimeError(f"non-contiguous numbered RGB images under {path}")
    return min(stems), max(stems), len(stems)


def arctic_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    data = root / "data"
    metadata = json.loads((data / "meta/misc.json").read_text())
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    test_candidates: list[dict[str, Any]] = []
    for mano_path in sorted((data / "raw_seqs").glob("s*/*.mano.npy")):
        subject = mano_path.parent.name
        name = mano_path.name.removesuffix(".mano.npy")
        sequence = f"{subject}/{name}"
        annotation = np.load(mano_path, allow_pickle=True).item()
        annotation_frames = int(len(annotation["left"]["rot"]))
        image_offset = int(metadata[subject]["ioi_offset"])
        image_first, image_last, image_count = _numeric_image_span(data / "images" / sequence / "0")
        available_start = max(0, image_first - image_offset)
        available_stop = min(annotation_frames, image_last - image_offset + 1)
        valid_start = available_start + ARCTIC_BOUNDARY_MARGIN
        valid_stop = available_stop - ARCTIC_BOUNDARY_MARGIN
        if valid_stop - valid_start < FRAMES:
            continue
        common = {
            "dataset": "arctic",
            "root": str(root),
            "sequence": sequence,
            "recording_id": sequence,
            "split_group": f"subject:{subject}",
            "num_frames": annotation_frames,
            "frames": FRAMES,
            "image_offset": image_offset,
            "image_first": image_first,
            "image_last": image_last,
            "image_count": image_count,
            "usable_start": valid_start,
            "usable_stop": valid_stop,
            "capabilities": list(FULL_CAPABILITIES),
        }
        maximum = valid_stop - FRAMES
        if subject == "s05":
            for start in range(valid_start, maximum + 1, FRAMES):
                test_candidates.append({**common, "split": "test", "start_frame": start})
        elif subject == "s04":
            output["val"].append(
                {
                    **common,
                    "split": "val",
                    "start_frame": fixed_start(sequence, valid_start, maximum),
                }
            )
        else:
            output["train"].append(
                {
                    **common,
                    "split": "train",
                    "start_min": valid_start,
                    "start_max": maximum,
                }
            )
    test_candidates.sort(
        key=lambda item: score(
            "arctic-test-segment", f"{item['recording_id']}:{item['start_frame']}"
        )
    )
    if len(test_candidates) < 291:
        raise RuntimeError(
            f"ARCTIC provides only {len(test_candidates)} clean non-overlapping test windows"
        )
    output["test"] = sorted(
        test_candidates[:291], key=lambda item: (item["recording_id"], item["start_frame"])
    )
    return output


def _mask_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid HOT3D mask value {value!r}")
    return normalized == "true"


def hot3d_mask(path: Path) -> list[tuple[int, bool]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            (int(row["timestamp[ns]"]), _mask_value(row["mask"]))
            for row in csv.DictReader(handle)
            if row["stream_id"] == HOT3D_STREAM and int(row["timestamp[ns]"]) > 0
        ]


def hot3d_valid_ranges(root: Path) -> tuple[int, list[tuple[int, int]], dict[str, int]]:
    base = hot3d_mask(root / "masks/mask_hand_pose_available.csv")
    if len({timestamp for timestamp, _ in base}) != len(base):
        raise RuntimeError(f"duplicate HOT3D RGB timestamps in {root}")
    masks = {
        name: dict(hot3d_mask(root / "masks" / f"{name}.csv")) for name in HOT3D_REQUIRED_MASKS
    }
    valid = [
        all(values.get(timestamp, False) for values in masks.values()) for timestamp, _ in base
    ]
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate([*valid, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= FRAMES:
                ranges.append((start, index))
            start = None
    stats = {
        "total_rgb_frames": len(base),
        "all_required_masks_true": sum(valid),
        "eligible_frames_in_long_runs": sum(stop - start for start, stop in ranges),
        "eligible_runs": len(ranges),
    }
    return len(base), ranges, stats


def _hot3d_common(
    path: Path,
    participant: str,
    frame_count: int,
    stats: dict[str, int],
) -> dict[str, Any]:
    return {
        "dataset": "hot3d",
        "recording_root": str(path),
        "recording_id": path.name,
        "participant_id": participant,
        "split_group": f"participant:{participant}",
        "num_frames": frame_count,
        "frames": FRAMES,
        "required_masks": list(HOT3D_REQUIRED_MASKS),
        "mask_stats": stats,
        "capabilities": list(FULL_CAPABILITIES),
    }


def hot3d_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Select valid windows from labelled Aria recordings with disjoint holdouts."""

    recordings: list[tuple[Path, int, str, list[tuple[int, int]], dict[str, int]]] = []
    labelled_recordings = 0
    labelled_participants: set[str] = set()
    for path in sorted(root.glob("P*_*")):
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("headset") != "Aria" or not metadata.get("have_hand_object_pose_gt", False):
            continue
        labelled_recordings += 1
        labelled_participants.add(str(metadata["participant_id"]))
        frame_count, ranges, stats = hot3d_valid_ranges(path)
        if ranges:
            recordings.append((path, frame_count, str(metadata["participant_id"]), ranges, stats))
    if labelled_recordings != 136:
        raise RuntimeError(
            f"expected 136 labelled HOT3D Aria recordings, got {labelled_recordings}"
        )
    participants = sorted(
        labelled_participants,
        key=lambda value: score("hot3d-participant-split", value),
    )
    if len(participants) != 9:
        raise RuntimeError(f"expected 9 labelled HOT3D Aria participants, got {participants}")
    test_participants = set(participants[:2])
    val_participants = {participants[2]}
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    test_candidates: list[dict[str, Any]] = []
    for path, frame_count, participant, ranges, stats in recordings:
        if participant in test_participants:
            split = "test"
        elif participant in val_participants:
            split = "val"
        else:
            split = "train"
        common = _hot3d_common(path, participant, frame_count, stats)
        if split == "train":
            output["train"].append(
                {
                    **common,
                    "split": split,
                    "valid_ranges": [list(value) for value in ranges],
                }
            )
        elif split == "val":
            output["val"].append(
                {
                    **common,
                    "split": split,
                    "start_frame": fixed_start_in_ranges(path.name, ranges),
                }
            )
        else:
            for start, stop in ranges:
                for frame_start in range(start, stop - FRAMES + 1, FRAMES):
                    test_candidates.append({**common, "split": split, "start_frame": frame_start})
    test_candidates.sort(
        key=lambda item: score(
            "hot3d-test-segment", f"{item['recording_id']}:{item['start_frame']}"
        )
    )
    if len(test_candidates) < 437:
        raise RuntimeError(
            f"HOT3D provides only {len(test_candidates)} clean non-overlapping test windows"
        )
    output["test"] = sorted(
        test_candidates[:437], key=lambda item: (item["recording_id"], item["start_frame"])
    )
    return output


def identities(records: list[dict[str, Any]], key: str) -> set[str]:
    return {str(record[key]) for record in records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/manifests/two_dataset_v2_clean"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=("arctic", "hot3d"),
        help="datasets to build; defaults to ARCTIC/HOT3D",
    )
    parser.add_argument(
        "--arctic-root",
        type=Path,
        default=Path("data/arctic"),
    )
    parser.add_argument("--hot3d-root", type=Path, default=Path("data/hot3d"))
    args = parser.parse_args()
    allowed_path(args.output)
    allowed_path(args.arctic_root)
    allowed_path(args.hot3d_root)
    selected = tuple(args.datasets)
    if len(set(selected)) != len(selected):
        raise ValueError("datasets must be unique")
    expected_files = {
        f"{dataset}_{split}.jsonl" for dataset in selected for split in ("train", "val", "test")
    }
    existing_files = {path.name for path in args.output.glob("*.jsonl")}
    unexpected_files = existing_files - expected_files
    if unexpected_files:
        raise RuntimeError(
            "output contains manifests for disabled datasets: " f"{sorted(unexpected_files)}"
        )

    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if "arctic" in selected:
        all_records["arctic"] = arctic_records(args.arctic_root)
    if "hot3d" in selected:
        all_records["hot3d"] = hot3d_records(args.hot3d_root)
    protocol_notes = {
        "arctic": (
            "Official subject ioi_offset maps annotation index to RGB filename; first/last "
            "10 available frames are excluded; s04 validation, s05 test, other subjects train."
        ),
        "hot3d": (
            "Inventory requires 136 labelled Aria recordings; only recordings with valid "
            "windows enter the participant-disjoint split. Only contiguous RGB "
            "ranges passing hand-pose, headset-pose, exposure and QA masks are eligible."
        ),
    }
    source_audit: dict[str, int] = {}
    if "arctic" in selected:
        source_audit["arctic_sequences_eligible"] = sum(
            len({row["recording_id"] for row in splits})
            for splits in all_records["arctic"].values()
        )
    if "hot3d" in selected:
        source_audit["hot3d_labelled_aria_recordings"] = 136
        source_audit["hot3d_recordings_with_clean_81_frame_run"] = len(
            {row["recording_id"] for splits in all_records["hot3d"].values() for row in splits}
        )
    report: dict[str, Any] = {
        "version": MANIFEST_SCHEMA,
        "seed": SEED,
        "frames_per_window": FRAMES,
        "selected_datasets": list(selected),
        "protocol_notes": {dataset: protocol_notes[dataset] for dataset in selected},
        "source_audit": source_audit,
        "datasets": {},
    }
    for dataset, splits in all_records.items():
        split_report: dict[str, Any] = {}
        identity_sets = {
            split: identities(records, "recording_id") for split, records in splits.items()
        }
        group_sets = {
            split: identities(records, "split_group") for split, records in splits.items()
        }
        recording_overlap = {
            "train_val": len(identity_sets["train"] & identity_sets["val"]),
            "train_test": len(identity_sets["train"] & identity_sets["test"]),
            "val_test": len(identity_sets["val"] & identity_sets["test"]),
        }
        group_overlap = {
            "train_val": len(group_sets["train"] & group_sets["val"]),
            "train_test": len(group_sets["train"] & group_sets["test"]),
            "val_test": len(group_sets["val"] & group_sets["test"]),
        }
        if any(recording_overlap.values()) or any(group_overlap.values()):
            raise RuntimeError(
                f"{dataset} split leakage: recordings={recording_overlap}, groups={group_overlap}"
            )
        for split, records in splits.items():
            count, sha = write_jsonl(args.output / f"{dataset}_{split}.jsonl", records)
            split_report[split] = {
                "rows": count,
                "recordings": len(identity_sets[split]),
                "split_groups": len(group_sets[split]),
                "sha256": sha,
                "evaluated_frames": count * FRAMES if split == "test" else None,
            }
        split_report["recording_overlap"] = recording_overlap
        split_report["split_group_overlap"] = group_overlap
        report["datasets"][dataset] = split_report
    report_path = args.output / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
