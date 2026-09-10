# HandPrism-Core model card

## Released models

Release `v0.3.1` contains **Core-Standard** and **Core-K-free**, each exported from its final step-20,000 checkpoint. Both use `handprism-core-r1`. HandPrism-Fusion source is included but no trained Fusion weights are released.

Each `.safetensors` file is approximately 670 MiB (702 MB), containing **755 trainable tensors / 175,503,289 parameters**, all retained at their original FP32 dtype. This includes all registered LoRA tensors, the patch embedding and registered diffusion head, decoder, ray head, and six small compatibility parameters from the hand module. Some registered backbone parameters are not reached by the clean-latent forward path; they are retained to preserve the exact loading schema, not counted as evidence that all parameters received gradients.

The two source checkpoint SHA-256 values, exported file sizes and SHA-256 digests are in [weights.json](../weights.json). Every exported key, dtype and tensor value was checked against the corresponding original checkpoint with exact equality. Export removes optimizer, scheduler, RNG, private configuration paths and unrelated metadata. It does not quantize tensors or add newly trained values.

These are **adapter/head inference bundles**, not standalone copies of the full frozen Wan model. They contain no training images, labels, clips, dataset manifests, predictions, MANO templates/meshes/blend shapes, or MANO model files.

## Training data and modes

Only **ARCTIC and HOT3D** were used, with dataset sampling weights 0.4375 and 0.5625. No EgoDex data are used. Each solver trained for 20,000 optimizer steps on 8 GPUs, with 1 clip per GPU and accumulation of 8 (Standard) or 4 (K-free), for effective batches 64 and 32 respectively. The original workflow completed training and full testing for each solver.

Standard inference uses camera calibration. K-free inference estimates an effective camera from predicted rays and does not take ground-truth intrinsics as input. The architecture is always explicitly selected as `handprism-core`; solver-specific files and configurations are not interchangeable.

## Required assets

- Wan2.2-Fun-5B-Control, revision `b8bc1a65ab71d054ba4636dc0dac104aa4df2686`.
- VideoX-Fun, revision `6f3fb60dad9b6a60ff6f962e62cffa11cafb084b`.
- Separately licensed left/right MANO models and the declared runtime dependencies.

The bootstrap script pins and checks the base-model artifacts. Users must obtain MANO and any evaluation data under their respective licenses. See [NOTICE.md](../NOTICE.md) and [weight-use terms](../WEIGHTS_TERMS.md).

## Compatibility and evaluation limits

The public Core implementation preserves the Core query topology, direct-joint interpolation and geometric acceptance paths. The shared runtime has updated MANO label handling, FP32 geometry, per-clip loss normalization and explicit wrist/translation metrics. **Historical metrics are not presented as metrics recomputed by this public release.** Exact weight equality does not establish identical full BF16 outputs across runtime revisions.

This publication verifies tensor export and CPU loading/contracts. It does not provide a new full-Wan GPU accuracy benchmark, latency benchmark or full current-runtime evaluation. CPU and asset-loading checks are described in [VALIDATION.md](VALIDATION.md). No claim is made that Fusion is better than Core.

Released files are inference/evaluation only. They cannot restore the original optimizer or scheduler. New Core/Fusion training starts at step 0; native training checkpoints can only resume under matching architecture, config and run contracts.

## Intended use and limitations

Non-commercial research and evaluation of hand reconstruction in egocentric video, subject to applicable upstream terms. Not validated for medical decisions, identity profiling, surveillance or safety-critical control. Domain shift, occlusion, blur, rare hand poses and inaccurate camera assumptions can cause failures. Camera-space trajectories are not automatically world-space trajectories. There is no guarantee of real-time execution, robust stereo correction, or physically safe control.

Prepared-input fields and prediction format are in [MODELS.md](MODELS.md); complete training/testing commands are in [RUNNING.md](RUNNING.md). Source code is MIT; the trained bundles and required third-party assets are not blanket MIT-licensed.
