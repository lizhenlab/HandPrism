#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${project_root}/third_party" "${project_root}/models"

if [[ ! -d "${project_root}/third_party/VideoX-Fun/.git" ]]; then
  git clone https://github.com/aigc-apps/VideoX-Fun.git "${project_root}/third_party/VideoX-Fun"
fi
git -C "${project_root}/third_party/VideoX-Fun" fetch origin 6f3fb60dad9b6a60ff6f962e62cffa11cafb084b
git -C "${project_root}/third_party/VideoX-Fun" checkout --detach 6f3fb60dad9b6a60ff6f962e62cffa11cafb084b

python_bin="${project_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "missing ${python_bin}; create the project environment first" >&2
  exit 2
fi

model_dir="${project_root}/models/Wan2.2-Fun-5B-Control"
mkdir -p "${model_dir}"

# Prefer the ModelScope mirror when its CLI is installed. Verify byte identity
# against the pinned Hugging Face revision; a mutable mirror label alone does
# not identify the model files used by a run.
if [[ -x "${project_root}/.venv/bin/modelscope" ]]; then
  "${project_root}/.venv/bin/modelscope" download \
    --model PAI/Wan2.2-Fun-5B-Control \
    --include config.json diffusion_pytorch_model.safetensors Wan2.2_VAE.pth \
    --local_dir "${model_dir}"
else
  DREAMHAND_MODEL_DIR="${model_dir}" "${python_bin}" - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alibaba-pai/Wan2.2-Fun-5B-Control",
    revision="b8bc1a65ab71d054ba4636dc0dac104aa4df2686",
    allow_patterns=[
        "config.json",
        "diffusion_pytorch_model.safetensors",
        "Wan2.2_VAE.pth",
    ],
    local_dir=os.environ["DREAMHAND_MODEL_DIR"],
)
PY
fi

DREAMHAND_MODEL_DIR="${model_dir}" "${python_bin}" - <<'PY'
import hashlib
import os
from pathlib import Path

expected = {
    "config.json":
        "dc20f8568e6b08121aa8e388c8cddba2ed42e9e5e57c7d75fbb6bf3b771cd018",
    "diffusion_pytorch_model.safetensors":
        "ace4718a7c87ee3e5606a68ab79142c4395e81aece76b8120bc886f0fbbe1d16",
    "Wan2.2_VAE.pth":
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
}
root = Path(os.environ["DREAMHAND_MODEL_DIR"])
for name, wanted in expected.items():
    digest = hashlib.sha256()
    with (root / name).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got != wanted:
        raise SystemExit(f"SHA-256 mismatch for {name}: {got} != {wanted}")
    print(f"verified {name}: {got}")
PY
