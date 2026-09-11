#!/usr/bin/env python3
"""Render GT joint overlays for visual camera/data-contract QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handprism.data import HandPrismWindowDataset


BONES = tuple(
    (0, base + 1) if offset == 0 else (base + offset, base + offset + 1)
    for base in (0, 4, 8, 12, 16)
    for offset in range(4)
)
COLORS = ((0, 230, 255), (255, 80, 200))


def render_sample(sample, output: Path) -> dict[str, object]:
    selected = (0, sample.video.shape[1] // 2, sample.video.shape[1] - 1)
    panels: list[Image.Image] = []
    for frame in selected:
        array = (
            sample.video[:, frame]
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        panel = Image.fromarray(np.asarray(array), mode="RGB")
        draw = ImageDraw.Draw(panel)
        height, width = panel.height, panel.width
        points = sample.joints_2d[frame].cpu()
        valid = sample.valid_joints_2d[frame].cpu()
        for side, color in enumerate(COLORS):
            pixels = [(float(point[0]) * width, float(point[1]) * height) for point in points[side]]
            for first, second in BONES:
                if bool(valid[side, first] and valid[side, second]):
                    draw.line((pixels[first], pixels[second]), fill=color, width=2)
            for joint, point in enumerate(pixels):
                if bool(valid[side, joint]):
                    radius = 3 if joint else 5
                    draw.ellipse(
                        (
                            point[0] - radius,
                            point[1] - radius,
                            point[0] + radius,
                            point[1] + radius,
                        ),
                        fill=color,
                    )
        label = f"{sample.dataset}  frame {int(sample.frame_indices[frame])}"
        draw.rectangle((4, 4, 8 + 8 * len(label), 26), fill=(0, 0, 0))
        draw.text((8, 8), label, fill=(255, 255, 255))
        panels.append(panel)
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
    )
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {
        "dataset": sample.dataset,
        "recording_id": sample.recording_id,
        "frames": [int(sample.frame_indices[index]) for index in selected],
        "output": str(output),
        "valid_joint_counts": [
            int(sample.valid_joints_2d[index].sum()) for index in selected
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifests/two_dataset_v2_clean"),
    )
    parser.add_argument("--mano-model", type=Path, default=Path("assets/body_models/mano"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    mano_model = args.mano_model if args.mano_model.is_absolute() else root / args.mano_model
    output_root = args.output if args.output.is_absolute() else root / args.output
    report = []
    for dataset in ("arctic", "hot3d"):
        data = HandPrismWindowDataset(
            manifest_root / f"{dataset}_{args.split}.jsonl",
            mano_model_path=mano_model,
            training=args.split == "train",
        )
        index = (0, 260820308) if args.split == "train" else 0
        report.append(render_sample(data[index], output_root / f"{dataset}_{args.split}.png"))
    payload = {"format": "handprism-data-overlay-v1", "samples": report}
    (output_root / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
