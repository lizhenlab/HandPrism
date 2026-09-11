"""Public descriptions, neutral entrypoints and data-scope regression checks."""

import ast
import importlib
import io
import json
from pathlib import Path
import re
import tokenize

import pytest

from handprism.architectures import ARCHITECTURES, FUSION, architecture_spec
from handprism.data.mixture import DEFAULT_DATASET_WEIGHTS, DatasetMixture
from handprism.decoder import HandPrismDecoder
from handprism.ray import RayHead
from scripts.inspect_model import inspect_architecture
from scripts.train import implementation_notes, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["train", "evaluate", "run_pipeline", "check_readiness", "build_manifests"])
def test_canonical_entrypoints_are_importable(entry):
    canonical = importlib.import_module(f"scripts.{entry}")
    assert callable(canonical.main)
    assert Path(canonical.__file__).name == f"{entry}.py"


def test_runtime_uses_descriptive_function_names():
    from handprism import lora, ray

    assert lora.configure_trainable_backbone.__name__ == "configure_trainable_backbone"
    assert ray.kfree_bearings.__name__ == "kfree_bearings"


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_parameter_report_counts_instantiated_components(architecture):
    report = inspect_architecture(architecture)
    decoder = HandPrismDecoder(architecture=architecture)
    ray = RayHead()
    assert report["architecture"] == architecture
    assert report["implementation_id"] == architecture_spec(architecture).implementation_id
    assert report["decoder_parameters"] == sum(p.numel() for p in decoder.parameters())
    assert report["ray_head_parameters"] == sum(p.numel() for p in ray.parameters()) == 9219
    assert report["decoder_and_ray_parameters"] == (
        report["decoder_parameters"] + report["ray_head_parameters"]
    )
    assert "excludes Wan" in report["scope"]


def test_all_active_configs_and_default_mixture_have_only_two_sources():
    assert DEFAULT_DATASET_WEIGHTS == {"arctic": 0.4375, "hot3d": 0.5625}
    mixture = DatasetMixture({"arctic": [1], "hot3d": [2]})
    assert mixture.names == ("arctic", "hot3d")
    active = []
    for path in (ROOT / "configs").glob("*.json"):
        config = json.loads(path.read_text())
        if config.get("deprecated"):
            continue
        checked = load_config(path, architecture=config["architecture"])
        assert checked["dataset_weights"] == DEFAULT_DATASET_WEIGHTS
        assert set(checked["dataset_roots"]) == set(DEFAULT_DATASET_WEIGHTS)
        active.append(path)
    assert len(active) >= 6


def test_retired_metadata_cannot_be_used_as_a_training_config():
    path = ROOT / "configs/components.json"
    value = json.loads(path.read_text())
    assert value["deprecated"] is True
    assert "datasets" not in value
    with pytest.raises(ValueError, match="training config is missing"):
        load_config(path, architecture=FUSION)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("solver", ["standard", "kfree"])
def test_generated_implementation_notes_describe_actual_settings(architecture, solver):
    stem = architecture.replace("-", "_")
    config = load_config(ROOT / f"configs/{stem}_{solver}.json", architecture=architecture)
    notes = implementation_notes(config)
    assert "ARCTIC and HOT3D" in notes[0]
    assert "8 attention heads and FFN width 1536" in notes[2]
    if solver == "kfree":
        assert config["kfree_camera_fit"]["target"] in notes[-1]
        assert config["kfree_camera_fit"]["assumption_note"] in notes[-1]


def test_active_descriptions_do_not_claim_external_method_equivalence():
    # Inspect comments and string constants, not compatibility identifiers or
    # immutable archive contents. Legitimate references remain in REFERENCES.md.
    forbidden = re.compile(
        r"\b(?:reproduction|reproduce|reproducing|paper|appendix|undisclosed)\b"
        r"|复现|忠实还原|论文实现",
        re.IGNORECASE,
    )
    violations = []
    for directory in (ROOT / "src", ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            source = path.read_text()
            tree = ast.parse(source)
            texts = [
                (node.lineno, node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            ]
            texts.extend(
                (token.start[0], token.string)
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            )
            violations.extend(
                f"{path.relative_to(ROOT)}:{line}" for line, value in texts
                if forbidden.search(value)
            )
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md"),
                 *(ROOT / "configs").glob("*.json")]:
        if forbidden.search(path.read_text()):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations
