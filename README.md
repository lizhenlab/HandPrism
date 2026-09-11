# HandPrism

**Geometry-aware 3D hand motion reconstruction from egocentric video.**

HandPrism 从第一人称 RGB 视频片段恢复左右手的三维姿态、形状和相机空间位置，结合视频时序特征、MANO 手部模型与射线几何，提供训练、验证、测试和预测导出工具。

## 模型

| 模型 | 必填架构参数 | 特点 | 训练权重 |
| --- | --- | --- | --- |
| HandPrism-Core | `handprism-core` | 各 query 独立时间交互、bearing 几何求解路径 | Standard / K-free 各 20,000 step |
| HandPrism-Fusion | `handprism-fusion` | 时空联合交互，可选原始 RGB 局部精修、关节引导 MANO、独立位置先验与稳健几何 | 尚未完整训练 |

Standard 使用输入相机标定；K-free 推理不需要输入相机内参。相机模式与模型架构是两个独立选择。训练、推理、评测必须手动指定 `--architecture`，没有默认架构，也不按文件名或参数形状猜测。

Fusion 指时间和查询信息融合，**不是新的双目输入分支**。当前推理入口接受预处理 RGB 片段，不自动拆分双目或直接处理 MP4。

### Fusion r4

当前开发实现为 `handprism-fusion-r4`；`r4` 是实现修订号，架构参数仍为 `handprism-fusion`，不是第三套模型或已训练权重版本。六项候选模块可独立或组合消融；完整配置不代表全部模块已经证明有效。B0 保留全局基础计算路径，Core 的默认行为、配置及发布权重保持不变。

本次训练前修订：

- **独立深度监督**：有效腕点的正相机 Z 提供 `log_depth` 监督，即便几何求解回退到位置先验，深度头仍有训练信号。
- **安全处理无效标注**：在非线性计算前屏蔽无效 GT，避免 NaN 污染反向传播；有效 GT 或模型预测非有限时直接报错。
- **离散 ROI 课程**：按片段/手选择完整 GT 框或预测框，避免框中心插值裁到背景；验证、测试和推理不使用 GT 框。
- **区分快速与完整验证**：同一验证 recording 内冻结最多 12 个非重叠窗口；快速验证每数据集全局 16 段，完整验证遍历全部冻结窗口，只有完整验证用于选择 best。

训练目标与验证索引发生变化，r3 断点不能直接续训为 r4。详细设计、开关及消融方法见 [Fusion 实验指南](docs/FUSION_R4.md)。

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

启用 Fusion `local_rgb` 时，还必须提供同帧、同视场的 `rgb_high` 原始细节图；不能从低分辨率 `video` 放大伪造。Core 不需要该字段。

## 训练与测试

训练、验证和测试**只允许 ARCTIC + HOT3D**，采样权重为 **0.4375 / 0.5625**。不读取或使用 EgoDex。原始数据、标注与数据清单不随源码分发；请合法取得数据，并按 [运行说明](docs/RUNNING.md) 构建清单、配置路径。

所采用数据版本的固定划分：

| 数据集 | 训练 recording | 验证 recording | 测试 recording | 测试片段 |
| --- | ---: | ---: | ---: | ---: |
| ARCTIC | 236 | 31 | 34 | 291 |
| HOT3D | 92 | 19 | 24 | 437 |

完整测试共 728 段、每段 81 帧。构建器会检查上述数量，不接受用较小数据子集冒充完整测试。测试集不用于选择配置。

先按 [运行说明](docs/RUNNING.md#数据准备) 准备基础清单，再为 Fusion/B0 构建同一份独立 r4 索引并检查 readiness：

```bash
bash scripts/v3_python.sh scripts/prepare_fusion_index.py \
  --source data/manifests/two_dataset_v2_clean \
  --output data/manifests/two_dataset_fusion_r4 \
  --mano-model assets/body_models/mano

bash scripts/v3_python.sh scripts/check_readiness.py \
  --architecture handprism-fusion \
  --configs configs/handprism_fusion_b0_standard.json configs/handprism_fusion_b0_kfree.json
```

索引构建只读取 train/val 几何，测试清单逐字节复制；不覆盖已有输出。readiness 通过后仍须完成真实片段短训、恢复及正式分辨率多卡验收，再冻结源码与配置启动正式实验。以下为 B0-Standard 训练入口：

```bash
bash scripts/v3_python.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train.py --architecture handprism-fusion \
  --config configs/handprism_fusion_b0_standard.json --run-dir runs/handprism_fusion_r4_b0_standard
```

新 Core / Fusion 实验从 step 0 开始。只有同架构、同配置、同源码契约的新训练 checkpoint 可以恢复。Core 权重不能改名用作 Fusion 权重。

先比较 B0 和单项改动，再训练入选组合；不要直接把六项同时启用后的变化归因于某一个模块。Standard/K-free 有效 batch 为 64/32，等 step 不等于等样本曝光。

## 验证边界

Core 的两种模式已完成原训练与测试；当前公共运行层包含数据、FP32 几何和指标口径修正，因此**不能把历史指标直接当作当前代码的重新评测结果**。Fusion 尚无完整训练权重，也未证明优于 Core。

Fusion r4 最近一次验收（2026-09-11）：

- Mac 回归 **222 passed / 3 skipped**；Linux 回归 **223 passed / 2 skipped**，Linux 包含真实 MANO 测试。
- Standard/K-free 的 CUDA/BF16 手部检查、双卡梯度累积及参数/优化器状态恢复通过；ARCTIC/HOT3D 各五帧真实样本读取通过。
- Standard 的实际 Wan VAE/DiT 前后向检查通过，但只使用 **5×64×64 合成片段**；不是完整 Wan 八卡或正式分辨率验收。

本轮完整 r4 索引尚未生成；真实数据短训、**81 帧全分辨率八卡验收**、等曝光消融和完整测试仍待执行。上述检查不证明收敛、精度提升、实时性能或 SOTA。跳过项、检查范围和历史记录见 [验证记录](docs/VALIDATION.md)。

## 文档与许可

- [模型与输入](docs/MODELS.md) · [模型卡](docs/MODEL_CARD.md) · [运行说明](docs/RUNNING.md)
- [Python API 与命令入口](docs/API.md)
- [设计说明](docs/DEVELOPMENT.md) · [Fusion 升级](docs/FUSION_R4.md) · [评测口径](docs/EVALUATION.md) · [验证记录](docs/VALIDATION.md)
- [参考资料](docs/REFERENCES.md) · [第三方声明](NOTICE.md) · [权重使用说明](WEIGHTS_TERMS.md)

HandPrism 自有代码和文档采用 [MIT License](LICENSE)。第三方组件、预训练主干、MANO 和数据集分别遵循其原许可。训练权重不统一重授权为 MIT。Python 导入名和发行包名均为 `handprism`，项目类名统一使用 `HandPrism*`。
