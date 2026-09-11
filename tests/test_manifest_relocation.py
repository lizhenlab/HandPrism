from __future__ import annotations

import copy
import hashlib
import json

import pytest

from handprism.data.schema import MANIFEST_SCHEMA
from scripts.check_readiness import EXPECTED_CAPABILITIES, HOT3D_REQUIRED_MASKS, required_paths
from scripts.relocate_manifests import relocate, relocated_record
from scripts.train import manifest_report


def fixture_manifests(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    roots = {name: tmp_path / name for name in ("arctic", "hot3d")}
    for part in ("meta", "raw_seqs", "images"):
        (roots["arctic"] / "data" / part).mkdir(parents=True)
    roots["hot3d"].mkdir()
    report = dict(version=MANIFEST_SCHEMA, frames_per_window=81,
                  selected_datasets=["arctic", "hot3d"], datasets={})
    rows = {}
    for dataset in roots:
        report["datasets"][dataset] = {}
        for split in ("train", "val", "test"):
            identity = f"s01/{split}" if dataset == "arctic" else f"P0001_{split}"
            row = dict(dataset=dataset, split=split, recording_id=identity, split_group=split,
                       frames=81, num_frames=100, capabilities=sorted(EXPECTED_CAPABILITIES[dataset]))
            if split == "train":
                row.update(start_min=0, start_max=19)
            else:
                row["start_frame"] = 0
            if dataset == "arctic":
                row.update(root=str(tmp_path / "absent_arctic"), sequence=identity, image_offset=1)
            else:
                row.update(recording_root=str(tmp_path / "absent_hot3d" / identity),
                           required_masks=sorted(HOT3D_REQUIRED_MASKS))
            rows[f"{dataset}_{split}.jsonl"] = row
            payload = (json.dumps(row) + "\n").encode()
            (source / f"{dataset}_{split}.jsonl").write_bytes(payload)
            report["datasets"][dataset][split] = dict(rows=1, recordings=1, split_groups=1,
                sha256=hashlib.sha256(payload).hexdigest())
            mapped = relocated_record(row, roots)
            for path in required_paths(mapped):
                path.parent.mkdir(parents=True, exist_ok=True)
                if dataset == "arctic" and path.name == "0":
                    path.mkdir()
                else:
                    path.touch()
    (source / "split_report.json").write_text(json.dumps(report))
    return source, roots, rows


def test_relocation_preserves_every_non_path_field_and_source_bytes(tmp_path):
    source, roots, rows = fixture_manifests(tmp_path)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / "relocated"
    report = relocate(source, output, roots["arctic"], roots["hot3d"])
    assert set(path.name for path in output.glob("*.jsonl")) == set(rows)
    for name, original_row in rows.items():
        loaded = json.loads((output / name).read_text())
        field = "root" if original_row["dataset"] == "arctic" else "recording_root"
        assert loaded.pop(field) != original_row[field]
        expected = copy.deepcopy(original_row)
        expected.pop(field)
        assert loaded == expected
    assert original == {path.name: path.read_bytes() for path in source.iterdir()}
    assert report["relocation"]["required_paths_checked"] == 33
    assert manifest_report(output, ("arctic", "hot3d"))["split_report"] == report
    with pytest.raises(ValueError, match="new directory"):
        relocate(source, output, roots["arctic"], roots["hot3d"])


def test_missing_data_does_not_publish_a_manifest(tmp_path):
    source, roots, rows = fixture_manifests(tmp_path)
    required_paths(relocated_record(rows["hot3d_train.jsonl"], roots))[0].unlink()
    output = tmp_path / "missing-output"
    with pytest.raises(FileNotFoundError, match="required paths missing"):
        relocate(source, output, roots["arctic"], roots["hot3d"])
    assert not output.exists()


@pytest.mark.parametrize("identity", ["../outside", "/absolute", "..", "", "a\\b"])
def test_recording_identity_cannot_escape_new_root(identity, tmp_path):
    row = dict(dataset="hot3d", split="train", recording_id=identity,
               recording_root=str(tmp_path / "recording"))
    with pytest.raises(ValueError, match="recording identity"):
        relocated_record(row, {"hot3d": tmp_path})


def test_source_hash_mismatch_is_not_repaired_by_relocation(tmp_path):
    source, roots, _ = fixture_manifests(tmp_path)
    path = source / "arctic_train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="does not match"):
        relocate(source, tmp_path / "output", roots["arctic"], roots["hot3d"])
