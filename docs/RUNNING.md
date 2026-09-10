# 运行说明

## 环境与资产

在新的环境中安装，不要改写其他实验共用的虚拟环境：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,backbone,data,mano]'
bash scripts/bootstrap_backbone.sh
```

项目最低 Python 3.10；发布 CPU 检查环境为 Python 3.11 / PyTorch 2.6。完整主干建议使用有 CUDA 的 Linux 环境。第三方依赖可能需要与具体驱动、PyTorch 对齐，基础 CPU 测试不保证完整环境已就绪。

`scripts/v3_python.sh` 使用当前仓库 `.venv` 并显式选择当前源码，不修改共享解释器的 editable 安装。脚本名是兼容入口，与模型架构选择无关。

单独合法取得 MANO 左右手模型后放入 `assets/body_models/mano/`。主干脚本将 VideoX-Fun 固定至 `6f3fb60dad9b6a60ff6f962e62cffa11cafb084b`，Wan 固定至 `b8bc1a65ab71d054ba4636dc0dac104aa4df2686`，并校验文件哈希。未随仓库附送数据、标注、MANO 或完整 Wan 主干。

## 数据准备

只使用 ARCTIC/HOT3D。六份配置的 `dataset_roots` 使用相对路径 `data/arctic`、`data/hot3d`；按自己的数据位置修改或链接。ARCTIC 根目录应包含 `data/meta`、`data/raw_seqs`、`data/images`；HOT3D 根目录应包含对应 `P*_*` recording、VRS、标注和 masks。

```bash
bash scripts/v3_python.sh scripts/build_manifests.py \
  --datasets arctic hot3d --arctic-root data/arctic --hot3d-root data/hot3d \
  --output data/manifests/two_dataset_v2_clean
```

构建器针对 README 中声明的数据版本和完整数量，不是任意数据子集的通用划分器。准备好完整数据后再执行；不覆盖已经冻结的清单。划分和连续有效区间规则见代码，数据的许可和访问手续由使用者自行办理。

## 训练、验证、完整测试

先验收 CUDA/BF16 前后向、多卡梯度累积、短训、验证和恢复，再冻结配置启动完整实验。下面命令不会由安装过程自动执行：

```bash
bash scripts/v3_python.sh scripts/check_readiness.py --architecture handprism-fusion \
  --configs configs/handprism_fusion_standard.json configs/handprism_fusion_kfree.json
bash scripts/v3_python.sh scripts/run_pipeline.py --architecture handprism-fusion
```

顺序为 readiness → train_standard（20,000 step）→ evaluate_standard → train_kfree（20,000 step）→ evaluate_kfree → complete。每 500 step 先验证再保存 checkpoint。不同架构分别使用 `runs/handprism_{core,fusion}_{full,standard,kfree}`，不得混用 state、checkpoint 和 metrics。

独立训练示例见 [README](../README.md)。Core 完整测试示例：

```bash
bash scripts/v3_python.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/evaluate.py --architecture handprism-core \
  --config configs/handprism_core_standard.json \
  --checkpoint models/handprism-core/handprism-core-standard-step020000.safetensors \
  --released-weights --output runs/core_standard_evaluation
```

完整测试必须包含 728 段（ARCTIC 291、HOT3D 437）；subset 诊断不能标记为完整测试。评测口径见 [EVALUATION](EVALUATION.md)。新训练与当前发布权重的边界见 [模型卡](MODEL_CARD.md)。

推理/训练/测试输出目录必须在当前仓库 `runs/` 内；入口拒绝覆盖既有推理结果和跨目录软链接。其他诊断工具有独立输出选项。

## CPU 测试

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  bash scripts/v3_python.sh -m pytest -q -p no:cacheprovider
```

有合法 MANO 资产时可额外设置 `MANO_MODEL_PATH` 为其目录。未提供时对应测试跳过；两项可选历史源码对齐测试也需要未随公共仓库分发的归档。必须如实报告通过与跳过数。权重导出工具仅供维护者使用，需 PyTorch 2.6+ / NumPy 2.x；普通推理直接使用 safetensors。

## 资源

正式配置使用 8 张 GPU。Standard 每卡 1 clip、累积 8 次；K-free 每卡 1 clip、累积 4 次。无 checkpoint 时按每份约 1.6 GiB 估算，一个架构的两种 solver 共 80 份约 128 GiB，另保留 80 GiB，完整流程预算约 208 GiB；不包含另行下载的主干和数据。

只检查资源，不抢占 GPU、不杀进程、不自动清理其他实验。监控应以实际运行目录和架构为准；当前公共仓库不包含个人定时任务或服务器凭据。
