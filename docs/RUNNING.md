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

只使用 ARCTIC/HOT3D。有效配置的 `dataset_roots` 使用相对路径 `data/arctic`、`data/hot3d`；按自己的数据位置修改或链接。ARCTIC 根目录应包含 `data/meta`、`data/raw_seqs`、`data/images`；HOT3D 根目录应包含对应 `P*_*` recording、VRS、标注和 masks。

```bash
bash scripts/v3_python.sh scripts/build_manifests.py \
  --datasets arctic hot3d --arctic-root data/arctic --hot3d-root data/hot3d \
  --output data/manifests/two_dataset_v2_clean
```

构建器针对 README 中声明的数据版本和完整数量，不是任意数据子集的通用划分器。准备好完整数据后再执行；不覆盖已经冻结的清单。划分和连续有效区间规则见代码，数据的许可和访问手续由使用者自行办理。

### 已有数据搬到新的挂载点

软链接或 `dataset_roots` 配置不会自动覆盖清单每一行中保存的绝对路径。已有冻结划分时，不需要重新解压、复制数据或重新随机划分；可生成只调整根路径的新清单：

```bash
bash scripts/v3_python.sh scripts/relocate_manifests.py \
  --source data/manifests/two_dataset_v2_clean \
  --output data/manifests/two_dataset_relocated \
  --arctic-root data/arctic --hot3d-root data/hot3d
```

ARCTIC 根目录必须直接包含 `data/{meta,raw_seqs,images}`，不一定是解压挂载的最外层。工具先验证旧清单哈希及数据契约，再检查新路径下每个 recording 所需文件；只更新 `root` / `recording_root` 和对应文件摘要，保持记录顺序、划分、窗口、有效区间、能力标识与其他字段不变。它不解码 train/val/test 图像或几何，也不搬移原始数据。输出必须不存在；原清单只读保留，新报告记录来源摘要。中断后的不完整输出不能通过 readiness。

将本机配置副本放在已忽略的 `configs/local/`，令 Core 的 `manifests` 指向新清单，并填写对应数据根目录。公开 Core 配置与权重身份保持不变。随后运行正常 readiness，不能用 `--skip-data-files` 作为数据迁移完成的证据。

Fusion 仍须在路径更新后的基础清单上单独生成下述 r4 索引；不能把普通清单改名来冒充困难窗口索引。本机 Fusion/B0 配置应共同指向同一份完整 r4 索引。

Fusion r4 在原划分上另建索引；Core 不使用新索引和增强。该步骤读取 train/val 几何、逐字节复制 test 清单，不读取 test RGB 或 test 标注几何、不改变划分：

```bash
bash scripts/v3_python.sh scripts/prepare_fusion_index.py \
  --source data/manifests/two_dataset_v2_clean \
  --output data/manifests/two_dataset_fusion_r4 --mano-model assets/body_models/mano
```

输出目录须不存在；索引版本为 2，所有六份清单重新校验 SHA-256。train 保存候选困难窗口，val 扩展为每 recording 最多 12 个固定、不重叠、兼顾时间与分层的窗口（`--val-windows-per-recording`），同一 held-out recording/participant 不换组。HOT3D 仅从 val recording 的原 required masks 恢复有效区间并核对统计；test 保持字节相同。旧版单窗口索引会被拒绝，不能只修改版本号。失败保留临时目录供诊断，不覆盖旧数据。耗时取决于标注数量与 VRS 元数据 I/O。未准备索引时不要启动完整 Fusion 候选。

## 训练、验证、完整测试

先验收 CUDA/BF16 前后向、多卡梯度累积、短训、验证和恢复，再冻结配置启动完整实验。下面命令不会由安装过程自动执行：

```bash
bash scripts/v3_python.sh scripts/check_readiness.py --architecture handprism-fusion \
  --configs configs/handprism_fusion_standard.json configs/handprism_fusion_kfree.json
bash scripts/v3_python.sh scripts/run_pipeline.py --architecture handprism-fusion
```

顺序为 readiness → train_standard（20,000 step）→ evaluate_standard → train_kfree（20,000 step）→ evaluate_kfree → complete。每 500 step 小验证、保存 checkpoint；Fusion 每 2,000 step 及最终 step 完整验证，只有完整验证能更新 `checkpoints/best.json`。完整流程仍固定评测 step 20000，不能把 best 与 final 的成绩混报。

Fusion 快速验证使用 `validation_clips_per_dataset=16`（全局预算，所有 GPU 合计），完整验证读取冻结 val 的所有窗口，两者日志均记录实际数量。Core 保留原每 rank 验证预算。训练日志包含独立 `loss/log_depth`、几何/先验融合比例、ROI 覆盖率与分支梯度诊断；loss 权重不等于梯度贡献。

Core 默认目录为 `runs/handprism_core_{full,standard,kfree}`，Fusion 为 `runs/handprism_fusion_r4_{full,standard,kfree}`。清单目录从两份配置共同推导；不一致、旧实现 control state 或其他架构目录会拒绝继续。不得混用 state、checkpoint 和 metrics。先做 B0 和短程单项消融，具体命令见 [Fusion 指南](FUSION_R4.md)；完整流程不是短程筛选入口。

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
