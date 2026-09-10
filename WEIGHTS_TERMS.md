# Trained-weight use and licensing scope

The HandPrism-Core release is provided for **non-commercial research and evaluation**. It is not covered by the repository's MIT source-code license and is not offered as a blanket commercial-use grant.

The models were trained on ARCTIC and HOT3D using a Wan video backbone and MANO geometry. Obtain and comply with all applicable upstream licenses and dataset/model agreements. The ARCTIC provider restricts use and redistribution of its supplied data and software; HOT3D and MANO have separate terms. This release does not redistribute those data or MANO assets and does not grant rights to them. Merely downloading these weights does not satisfy any separate asset-access requirements.

Upstream-derived Wan parameters retain applicable Apache-2.0 provisions and notices; see [NOTICE.md](NOTICE.md). Nothing here removes those upstream rights or conditions. HandPrism does not assert that every possible downstream use, redistribution, or commercial deployment has been cleared by third-party rights holders. Obtain any additional permission required for your use; do not treat the source MIT file as such permission.

Do not use the release for surveillance, identity profiling, safety-critical control, medical decisions, or other uses inconsistent with the applicable data/model terms. It has not been validated for those purposes. Accuracy, robustness, privacy protection and fitness for a particular purpose are not guaranteed. Results may fail under occlusion, motion blur, domain shift or inaccurate camera assumptions.

Include the component and dataset acknowledgements when reporting results. Review the [model card](docs/MODEL_CARD.md) for training scope, required external assets and evaluation limitations.
