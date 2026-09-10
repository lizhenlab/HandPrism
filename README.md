# HandPrism

**Geometry-aware 3D hand motion reconstruction from egocentric video.**

HandPrism 从第一人称 RGB 视频片段恢复左右手的三维姿态、形状和相机空间位置，结合视频时序特征、MANO 手部模型与射线几何，提供训练、验证、测试和预测导出工具。

## 模型

| 模型 | 必填架构参数 | 特点 | 训练权重 |
| --- | --- | --- | --- |
| HandPrism-Core | `handprism-core` | 各 query 独立时间交互、bearing 几何求解路径 | Standard / K-free 各 20,000 step |
| HandPrism-Fusion | `handprism-fusion` | 时间 × query 联合交互、坐标插值、像素域几何检查 | 尚未完整训练 |

Standard 使用输入相机标定；K-free 推理不需要输入相机内参。相机模式与模型架构是两个独立选择。训练、推理、评测必须手动指定 `--architecture`，没有默认架构，也不按文件名或参数形状猜测。

Fusion 指时间和查询信息融合，**不是新的双目输入分支**。当前推理入口接受预处理 RGB 片段，不自动拆分双目或直接处理 MP4。

## 已训练权重

从 [GitHub Release v0.3.1](https://github.com/lizhenlab/HandPrism/releases/tag/v0.3.1) 下载：

- [Core-Standard](https://github.com/lizhenlab/HandPrism/releases/download/v0.3.1/handprism-core-standard-step020000.safetensors)
- [Core-K-free](https://github.com/lizhenlab/HandPrism/releases/download/v0.3.1/handprism-core-kfree-step020000.safetensors)

两份均为最终训练权重的 **FP32、tensor-only 推理包**，不是随机初始化或占位文件。保留全部已保存的 trainable tensor，逐项验证与原 checkpoint 的键、dtype 和数值完全一致；移除优化器、RNG 和私人部署信息。它们不是完整的 Wan 主干，也不包含 MANO 资产，不能用来恢复原训练进度。

权重内容、依赖版本、SHA-256 和适用边界见 [模型卡](docs/MODEL_CARD.md) 与 [权重清单](weights.json)。**源码的 MIT 许可不代表训练权重、数据或 MANO 获得了 MIT 商用授权。**

## 安装与推理

完整模型建议在有 CUDA 的 Linux 环境中运行。先准备 Python 3.11 环境（项目最低 Python 3.10），在仓库根目录执行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,backbone,data,mano]'
bash scripts/bootstrap_backbone.sh
```

主干脚本下载并校验固定版本的 VideoX-Fun 与 Wan 文件；下载量较大。MANO 左右手模型须按原许可自行取得，放在 `assets/body_models/mano/`。不能用缺失的资产运行完整模型。

已安装 GitHub CLI 时，可下载两份权重并核验（也可通过上方链接手动下载）：

```bash
mkdir -p models/handprism-core
gh release download v0.3.1 --repo lizhenlab/HandPrism \
  --pattern '*.safetensors' --pattern SHA256SUMS \
  --dir models/handprism-core
(cd models/handprism-core && shasum -a 256 -c SHA256SUMS)
```

以 Core-K-free 推理为例：

```bash
bash scripts/v3_python.sh scripts/infer.py \
  --architecture handprism-core --config configs/handprism_core_kfree.json \
  --checkpoint models/handprism-core/handprism-core-kfree-step020000.safetensors \
  --released-weights --input /path/to/prepared_clip.npz --output runs/core_demo
```

输入 `video` 必须为 `uint8 [T,H,W,3]` RGB，T 为 1、5、…、81，H/W 为正的 32 倍数。Standard 另需对应输入尺寸的标定。完整字段见 [模型与输入说明](docs/MODELS.md)。输出为 `prediction.npz` 与 `provenance.json`；输出目录已存在时拒绝覆盖。

## 训练与测试

训练、验证和测试**只允许 ARCTIC + HOT3D**，采样权重为 **0.4375 / 0.5625**。不读取或使用 EgoDex。原始数据、标注与数据清单不随源码分发；请合法取得数据，并按 [运行说明](docs/RUNNING.md) 构建清单、配置路径。

所采用数据版本的固定划分：

| 数据集 | 训练 recording | 验证 recording | 测试 recording | 测试片段 |
| --- | ---: | ---: | ---: | ---: |
| ARCTIC | 236 | 31 | 34 | 291 |
| HOT3D | 92 | 19 | 24 | 437 |

完整测试共 728 段、每段 81 帧。构建器会检查上述数量，不接受用较小数据子集冒充完整测试。测试集不用于选择配置。

```bash
bash scripts/v3_python.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train.py --architecture handprism-fusion \
  --config configs/handprism_fusion_standard.json --run-dir runs/handprism_fusion_standard
```

新 Core / Fusion 实验从 step 0 开始。只有同架构、同配置、同源码契约的新训练 checkpoint 可以恢复。Core 权重不能改名用作 Fusion 权重。

## 验证边界

Core 的两种模式已完成原训练与测试；当前公共运行层包含数据、FP32 几何和指标口径修正，因此**不能把历史指标直接当作当前代码的重新评测结果**。Fusion 尚无完整训练权重，也未证明优于 Core。

CPU 回归覆盖架构选择、几何、采样、权重身份校验和片段推理。完整 Wan CUDA/BF16、多卡短训与当前评测链路仍需单独验收；不宣称实时性能或精度提升。详见 [验证记录](docs/VALIDATION.md)。

## 文档与许可

- [模型与输入](docs/MODELS.md) · [模型卡](docs/MODEL_CARD.md) · [运行说明](docs/RUNNING.md)
- [设计说明](docs/DEVELOPMENT.md) · [评测口径](docs/EVALUATION.md) · [验证记录](docs/VALIDATION.md)
- [参考资料](docs/REFERENCES.md) · [第三方声明](NOTICE.md) · [权重使用说明](WEIGHTS_TERMS.md)

HandPrism 自有代码和文档采用 [MIT License](LICENSE)。第三方组件、预训练主干、MANO 和数据集分别遵循其原许可。训练权重不统一重授权为 MIT。为保持接口兼容，Python 内部导入名仍为 `dreamhand`；项目名和发行包名为 **HandPrism / handprism**。
