# Third-party notices

The MIT license in this repository covers HandPrism-owned source code and documentation, not all materials required to run the system.

## Wan / VideoX-Fun

The trained release includes LoRA parameters, a modified patch embedding, and the registered head parameters associated with **Wan2.2-Fun-5B-Control**, provided by Alibaba PAI and the Wan/VideoX-Fun contributors. The full frozen backbone and VAE are not bundled.

- Model: https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control
- Model revision: `b8bc1a65ab71d054ba4636dc0dac104aa4df2686`
- Runtime: https://github.com/aigc-apps/VideoX-Fun
- Runtime revision: `6f3fb60dad9b6a60ff6f962e62cffa11cafb084b`

The upstream model card identifies Apache License 2.0. A copy of the pinned runtime's license is included at [licenses/VideoX-Fun-Apache-2.0.txt](licenses/VideoX-Fun-Apache-2.0.txt). Upstream copyrights and applicable license provisions remain in force for upstream-derived material. The HandPrism adaptation is not an upstream model release or endorsement.

## Data and hand models

- ARCTIC data, annotations, models and toolkit have separate [non-commercial terms](https://github.com/zc-alexfan/arctic/blob/master/LICENSE), including distribution restrictions on the materials supplied by ARCTIC. None of those data/model assets are bundled here.
- HOT3D data are subject to the provider's dataset agreement, distinct from the toolkit's Apache license: https://github.com/facebookresearch/hot3d#license
- MANO assets require a separate license: https://mano.is.tue.mpg.de/. The release does not contain MANO meshes, templates, blend shapes, regressors or model files.
- Dependencies such as PyTorch, NumPy, SMPL-X and camera/video tooling retain their own licenses.

Method and dataset acknowledgements are listed in [docs/REFERENCES.md](docs/REFERENCES.md). See [WEIGHTS_TERMS.md](WEIGHTS_TERMS.md) before using the trained weights. No source-code license grants rights to third-party data, personal likenesses, or model assets.
