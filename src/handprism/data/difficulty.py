"""Fixed GT-only difficulty strata and deterministic validation ordering."""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib

import torch

from .contract import HandPrismSample

FUSION_INDEX_VERSION = 2
VALIDATION_WINDOW_POLICY = "temporal-stratified-nonoverlap-v2"
STRATA = {"oos", "edge", "small", "fast", "occluded"}


def select_validation_windows(candidates: list[dict], maximum: int = 12) -> list[dict]:
    """Freeze per-recording coverage before training, never using predictions.

    Half the budget spans time uniformly; the remainder interleaves difficulty
    strata. Inputs must be nonoverlapping eligible windows from one recording.
    Short recordings keep all candidates rather than duplicating windows.
    """
    if type(maximum) is not int or maximum < 2 or not candidates:
        raise ValueError("validation requires candidates and maximum >= 2")
    candidates = sorted(candidates, key=lambda r: r["start_frame"])
    if any(b["start_frame"] < a["start_frame"] + a.get("frames", 81)
           for a, b in zip(candidates, candidates[1:])):
        raise ValueError("validation candidates must not overlap")
    if len(candidates) <= maximum:
        return candidates
    count = (maximum + 1) // 2
    chosen = {round(i * (len(candidates) - 1) / max(count - 1, 1)) for i in range(count)}
    for index in stratified_order(candidates):
        chosen.add(index)
        if len(chosen) == maximum:
            break
    return [candidates[i] for i in sorted(chosen)]


def validation_rank_limit(global_limit: int, rank: int, world: int) -> int | None:
    """Limit a disjoint stride-sharded global prefix, independent of GPU count."""
    if type(global_limit) is not int or global_limit < 0 or not 0 <= rank < world:
        raise ValueError("invalid global validation budget/rank")
    return None if global_limit == 0 else max(0, (global_limit - rank + world - 1) // world)


def validate_fusion_index(report: dict, records: dict[str, dict[str, list[dict]]],
                          fast_limit: int = 0) -> None:
    """Check frozen validation coverage and ranges without opening raw data."""
    index = report.get("fusion_index", {})
    if (index.get("version") != FUSION_INDEX_VERSION or index.get("test_used_for_design") is not False
            or index.get("validation_policy") != VALIDATION_WINDOW_POLICY):
        raise ValueError("requires Fusion index v2 with frozen multiwindow validation")
    for name in ("arctic", "hot3d"):
        if any(not isinstance(row.get("difficulty_windows"), list) for row in records[name]["train"]):
            raise ValueError("missing train-only difficulty windows")
        rows = records[name]["val"]
        if len(rows) <= fast_limit:
            raise ValueError("full validation must contain more windows than fast validation")
        groups = defaultdict(list)
        for row in rows:
            tags = row.get("validation_strata")
            if not isinstance(tags, list) or any(tag not in STRATA for tag in tags):
                raise ValueError("invalid frozen validation strata")
            start, frames = row["start_frame"], row["frames"]
            ranges = row.get("valid_ranges", [])
            if (frames != 81 or not ranges or not any(lo <= start and start + frames <= hi for lo, hi in ranges)
                    or any(lo < 0 or hi > row["num_frames"] or hi - lo < frames for lo, hi in ranges)):
                raise ValueError("validation window outside certified ranges")
            groups[row["recording_id"]].append(start)
        for starts in groups.values():
            ordered = sorted(starts)
            if any(b < a + 81 for a, b in zip(ordered, ordered[1:])):
                raise ValueError("duplicate/overlapping validation windows")
        if len(rows) <= len(groups):
            raise ValueError("validation index has not expanded beyond one window per recording")
        declared = index["validation"][name]
        if declared["windows"] != len(rows) or declared["recordings"] != len(groups):
            raise ValueError("validation coverage summary mismatch")
        if any(len(starts) > index.get("val_windows_per_recording", 12) for starts in groups.values()):
            raise ValueError("validation exceeds frozen per-recording budget")


def window_strata(sample: HandPrismSample) -> list[str]:
    uv = sample.joints_2d
    valid = sample.valid_joints_3d.bool() & sample.valid_hand[..., None]
    inside = valid & ((uv >= 0) & (uv < 1)).all(-1) & (sample.joints_camera[..., 2] > .01)
    labels = []
    scale = sample.image_size[[1, 0]]
    if (sample.valid_hand & ~inside.any(-1)).any():
        labels.append("oos")
    if (inside & (torch.minimum(uv, 1-uv) * scale < 32).any(-1)).any():
        labels.append("edge")
    lo = torch.where(inside[..., None], uv, torch.inf).amin(-2)
    hi = torch.where(inside[..., None], uv, -torch.inf).amax(-2)
    if (((hi-lo) * scale).norm(dim=-1) < 48).any():
        labels.append("small")
    if sample.timestamps is not None:
        dt = sample.timestamps.diff()
        valid_time = torch.isfinite(dt) & (dt > 1e-6) & (dt <= .15)
        speed = sample.joints_camera[..., 0, :].diff(dim=0).norm(dim=-1) / dt.clamp_min(1e-6)[:, None]
        valid_wrist = valid[1:, :, 0] & valid[:-1, :, 0] & valid_time[:, None]
        if ((speed > 1.) & valid_wrist).any():
            labels.append("fast")
    if sample.observed_valid is not None and sample.observed is not None:
        if (inside & sample.observed_valid & ~sample.observed).any():
            labels.append("occluded")
    return labels


def stratified_order(records: list[dict]) -> list[int]:
    """Interleave frozen difficulty/groups; each record appears exactly once.

    If difficulty metadata is unavailable, split-group interleaving is the
    explicit fallback. The same global order is sharded disjointly across ranks.
    """
    # Rotate recordings within each stratum/group so expanded windows from a
    # single recording do not dominate the fixed fast-validation prefix.
    grouped = defaultdict(lambda: defaultdict(deque))
    ordered = sorted(range(len(records)), key=lambda i: hashlib.sha256(
        f"{records[i].get('recording_id')}:{records[i].get('start_frame', 0)}".encode()).hexdigest())
    for index in ordered:
        record = records[index]
        strata = record.get("validation_strata", [])
        category = next((label for label in ("occluded", "oos", "fast", "small", "edge") if label in strata), "ordinary")
        grouped[(category, str(record.get("split_group", "unknown")))][str(record.get("recording_id"))].append(index)
    buckets = {}
    for key, recordings in grouped.items():
        queue = deque(recordings[name] for name in sorted(recordings, key=lambda name: hashlib.sha256(name.encode()).hexdigest()))
        buckets[key] = queue
    output = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                recording = buckets[key].popleft()
                output.append(recording.popleft())
                if recording:
                    buckets[key].append(recording)
    return output
