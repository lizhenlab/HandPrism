# 发布与开发验证记录

## Fusion r4 训练前修订验收（2026-09-11）

实现 ID 更新为 `handprism-fusion-r4`，序列化契约版本仍为 9。本轮修复深度头在独立先验回退时缺少监督、无效标注的非线性反向传播、快速/完整验证覆盖重合、ROI 教师框连续插值四项问题；只使用 ARCTIC/HOT3D。训练目标与验证协议改变，不能将 r3 断点自动作为 r4 续训。

- Mac 回归：**222 passed, 3 skipped in 18.63s**；Linux 回归：**223 passed, 2 skipped, 12 warnings in 36.87s**。Linux 运行了真实 MANO 检查；其余跳过项为未公开的历史源码 fixture，warning 来自既有依赖的弃用提示。
- 新增回归覆盖 Standard/K-free 冷启动深度梯度、NaN/±Inf 无效 GT 的有限前后向、有效 GT 与预测非有限时报错、相机 Z 深度目标、完整框离散 ROI 课程、跨卡数一致的快速验证预算、完整验证窗口的非重叠/边界/分层，以及索引生成与评分安全。索引生成端到端回归使用受控几何 fixture，不冒充完整真实索引构建。
- Core 前后逐位对照：Standard/K-free × FP32/CPU BF16 共 **4 组**，相同初始化及 Toy MANO 下，状态键和值、全部输出 tensor、原 loss 项及参数梯度完全一致。此检查不替代完整已发布模型推理。
- 服务器两份已发布 Core 权重 SHA-256 与固定发布记录一致。公开 Core 配置、原始清单及其测试划分未修改；路径版基础清单通过哈希审计，测试仍为 ARCTIC **291**、HOT3D **437** 段。

### 实际 GPU 与数据检查

环境为 Linux / Python 3.11 / PyTorch 2.6.0+cu124 / A800-SXM4-40GB。报告位于服务器工作区的 `runs/fusion_r4_checks_20260911/`，检查完成后无遗留 GPU 计算进程。

| 检查 | 结果 | 报告文件 |
| --- | --- | --- |
| Standard/K-free 手部模块 CUDA/BF16 | 各两次合成样本更新，梯度有限、深度头非零，训练参数/优化器状态恢复一致；约 2.65 / 2.62 s | `heads_standard.json`、`heads_kfree.json` |
| Standard/K-free 双卡手部模块、累积 2 次 | 各两次更新，跨 rank 梯度一致、深度头非零、状态恢复一致；约 2.68 / 2.87 s | `heads_ddp_standard.json`、`heads_ddp_kfree.json` |
| Standard 实际 Wan VAE/DiT + MANO | 两次更新，深度头、局部 CNN、patch embedding 及第 0/15 层 LoRA 等活跃分支获得有限非零梯度，状态恢复一致；约 15.36 s | `wan_standard.json` |
| ARCTIC/HOT3D 各 5 帧真实训练样本 | 原始细节图、全局图及时间契约检查通过；约 30.73 s | `real_data.json` |

Wan 检查只有 **5×64×64 合成片段**，峰值约 **11.94 GiB**；双卡检查只含手部模块，不含完整 Wan DDP。本轮没有重跑 K-free 的完整 Wan 检查，不能用上节历史 r3 结果代替。两次更新仅验证执行、梯度与恢复，不证明收敛、精度提升或正式分辨率吞吐。

真实 ARCTIC 片段的有效 3D 手部位于画面外，保留这些标注而不伪造在画面内监督；HOT3D 帧间隔约 33.29–33.33 ms。两者均没有伪造可靠逐关节遮挡标签。

### 验证索引与剩余训练门槛

只读核对实际验证记录的连续有效帧范围，按每 recording 最多 12 个非重叠窗口，预计 ARCTIC **31 个 recording / 277 个窗口**，HOT3D **19 个 recording / 221 个窗口**，合计 **498 个窗口**。这是范围预检估算，不是已发布索引；最终分层选择及数量必须以完整构建后通过审计的 `split_report.json` 为准。快速验证为每数据集全局 **16 个窗口**，不随 rank 数量倍增。

本轮已同步源码、配置、文档，并保留可恢复的修改前备份；未启动正式训练。服务器 Fusion/B0 本机配置指向待生成的独立 r4 索引。正式长训前仍需：

1. 从路径版基础清单生成并冻结完整 r4 train/val 索引，原样复制测试清单；完成 readiness、提交并冻结源码/配置。
2. 真实片段短训/过拟合，正式 **81 帧、全分辨率、8 GPU** 的有限值、峰值显存、吞吐和恢复验收。
3. B0/单项/组合的等曝光消融，仅用完整验证选择配置，再执行独立完整测试。

没有新增可发布的 Fusion 权重，也没有将本轮工程修正作为 SOTA 证据。设计和命令见 [Fusion r4 指南](FUSION_R4.md)。

## 数据根目录重新接入验收（2026-09-11）

- 在现有解压数据上生成独立的路径版清单，只调整 ARCTIC 的 `root` 与 HOT3D 的 `recording_root`，不复制或重新解压数据。ARCTIC 使用直接包含 `data/{meta,raw_seqs,images}` 的内层根目录。
- 六份新旧清单共 **1,106 条记录**逐行核对，记录顺序及所有非路径字段完全一致；原始清单与报告的 SHA-256 保持不变。测试集仍为 **728 段**，其中 ARCTIC **291**、HOT3D **437**，没有改变划分或窗口。
- **1,983 个**唯一所需文件/目录路径全部存在。Core Standard/K-free 的正常 readiness 检查通过，`ready=true`，没有使用跳过数据文件检查的选项；清单契约、模型资产及其哈希、依赖、GPU 和磁盘检查均通过。
- 机器专用配置放在 Git 忽略的 `configs/local/`，仅改变 `dataset_roots` 与 `manifests`，其余设置与对应模板一致。新清单和六份配置总计 **575,828 字节（约 0.55 MiB）**。公开 Core 配置及发布权重未修改；在新挂载点运行时应显式选择这些本机配置。
- Mac 回归：**184 passed, 3 skipped in 15.58s**；Linux 回归：**185 passed, 2 skipped, 12 warnings in 36.60s**。Linux 已运行真实 MANO 检查；跳过项为未公开的历史源码 fixture，warning 来自既有依赖的弃用提示。新增路径迁移测试覆盖不可覆盖输出、源哈希损坏、缺失数据与非法 recording 路径。

上述数据检查验证路径存在性与清单契约，不表示逐一解码了全部原始样本。本次没有启动训练、完整推理或完整评测。在该次路径接入验收时，Fusion/B0 的本机配置指向待独立生成的 r3 索引，不能用 Core 路径版清单替代；后续已更新为 r4，当前状态见上节。路径迁移用法见 [运行说明](RUNNING.md)。

## 目录迁移与独立启动验收（2026-09-11）

- 移动工作区后，重新定位虚拟环境入口并离线安装当前包；激活环境、`pip` 和从软链接入口运行脚本均通过。
- 主入口按脚本的真实位置解析源码和工具模块，避免依赖外部 `PYTHONPATH`。新增 **12 项**回归，在另一工作目录、清除 `PYTHONPATH` 后逐个运行 `--help`；此检查不启动训练或评测。
- Mac 回归：**175 passed, 3 skipped in 17.65s**；Linux 回归：**176 passed, 2 skipped, 12 warnings in 34.79s**。Linux 已运行真实 MANO 检查，跳过项和依赖警告含义与下节相同。
- 项目源码和环境迁移不改变模型结构、参数键或训练权重。旧源码与环境入口保留可恢复备份；现有清单、权重和历史训练产物没有原地改写。
- 代码回归不等于数据已就绪。目录迁移阶段的完整 readiness 检查中，清单哈希、两数据集划分、模型资产、依赖、GPU 和磁盘通过，但实际数据路径检查失败：当时核查的挂载位置仅提供压缩分片，旧清单记录的解压目录不可用。随后接入现有解压目录的结果见上节；这不等于 Fusion 索引及正式训练验收已经完成。

## Python API 与命名整理验收（2026-09-11）

源码包统一为 `handprism`，项目级类名统一为 `HandPrism*`，命令入口与元数据标识同步整理。本节仅验证命名迁移，不将检查结果用作模型精度或 SOTA 证据；下方 Fusion r3 的功能验收记录独立保留。

- Mac CPU 回归：**163 passed, 3 skipped in 8.11s**。跳过两项未公开的历史源码 fixture 和一项未配置本地 MANO 的测试。
- Linux 服务器回归：**164 passed, 2 skipped, 20 warnings in 17.24s**。真实 MANO 测试通过；两项跳过均为历史源码 fixture，warning 来自既有依赖的弃用提示。
- 命名前后逐位对照：Core/Fusion × Standard/K-free × FP32/CPU BF16 共 **8 组**全部通过。使用相同初始化和 Toy MANO，参数键、形状、初始值及全部输出 tensor 一致；Core 为 93 个状态 tensor，Fusion 为 135 个状态 tensor。该检查不替代完整 Wan GPU 推理。
- 两份实际已发布 Core safetensors 均通过固定 SHA-256 与元数据检查，每份 **755 个 tensor** 均有限；其中 **150 个手部 tensor**严格载入使用真实 MANO 的新命名模型，加载后数值逐项相等。权重文件没有改写。
- 既有双数据清单通过新入口只读检查：ARCTIC train/val/test 为 **236/31/291** 段，HOT3D 为 **92/19/437** 段；六份清单的哈希、数量、划分及数据集白名单检查通过。原始报告字节保持不变，内存报告规范化为当前 schema 并保留原格式摘要。
- `handprism-0.3.1` wheel 构建通过，发行内容只包含 `handprism` 包及其发行元数据，没有其他项目命名的 Python 包。
- 六项新增 namespace 回归覆盖公开导入、包版本、未知 schema 拒绝、清单只读性及哈希校验；已移除入口的测试由规范入口检查替换。

本次命名调整不需要重新训练，也没有启动训练或完整评测。外部脚本需更新导入并重新安装本地包；训练断点仍执行源码与配置契约检查。使用方式见 [API 与入口说明](API.md)。

## Fusion r3 开发验收（2026-09-11）

实现 ID 为 `handprism-fusion-r3`，契约版本 9。本次只改进 Fusion，并维持 ARCTIC/HOT3D 白名单；Core 默认配置、发布权重及权重清单未修改。以下结果不能用作模型精度或 SOTA 证据。

- Mac CPU 回归：**161 passed, 3 skipped in 11.55s**。跳过两项未公开的历史源码 fixture 和一项未配置本地合法 MANO 的测试。
- Linux 服务器回归：**162 passed, 2 skipped, 12 warnings in 24.44s**。真实 MANO 测试已运行；只跳过两项历史源码 fixture。warning 来自 MANO / NumPy / SciPy 等既有依赖的弃用提示。
- 静态检查：`git diff --check`、全部源码/脚本/测试的 Ruff 致命错误检查通过；新增 Python 模块的 Ruff `F` 检查通过。
- 27 项 r3 专项回归覆盖全部新增分支的双步 autograd、B0/Core 默认行为、共享初始化、ROI 半像素中心与坏 ROI 回退、禁止评测 GT ROI、时钟缺口、MANO 关节顺序、几何稳健性、OOS 绝对误差、质量/先验覆盖率、困难窗口边界、独立消融、固定曝光和 loss 加权账目。

### 实际 GPU 与数据检查

环境为 Linux / Python 3.11 / PyTorch 2.6.0+cu124 / NVIDIA A800-SXM4-40GB。各项使用新建隔离工作区及新报告文件，没有恢复或覆盖 Core/Fusion 历史训练产物。

| 检查 | 结果 | 报告文件 |
| --- | --- | --- |
| ARCTIC/HOT3D 各 5 帧真实训练样本 | 通过；约 3.09 s；原始细节图与全局图对齐，timestamps 有明确来源 | `real_data.json` |
| Standard/K-free 手部模块 CUDA/BF16 | 各两次合成样本更新，finite、训练参数/优化器状态恢复通过；约 3.42 / 3.40 s | `heads_standard.json`、`heads_kfree.json` |
| Standard 双卡手部模块、累积 2 次 | 两次更新，梯度跨 rank 一致，状态恢复通过；约 2.65 s | `ddp_accum_standard.json` |
| K-free 双卡手部模块、累积 2 次 | 同上，并确认新增分支收到非零梯度；约 3.12 s | `ddp_accum_kfree_gradients.json` |
| Standard 实际 Wan VAE/DiT + MANO | 两次更新；全部活跃分支、patch embedding、第 0/15 层 LoRA 有非零有限梯度，状态恢复通过；约 16.70 s | `wan_standard_gradients.json` |
| K-free 实际 Wan VAE/DiT + MANO | 同上；约 15.42 s | `wan_kfree_gradients.json` |

真实样本：ARCTIC 全局 `[3,5,480,672]`、细节 `[3,5,1006,1408]`，release 帧号 / 30 Hz；HOT3D 全局 `[3,5,480,480]`、细节 `[3,5,1408,1408]`，RGB TIME_CODE 时间差约 33.29–33.33 ms。两者均不伪造可靠遮挡标注。ARCTIC 抽中的起始片段手部出画，但有效 3D 标注保留。

Wan 检查只有 **5×64×64 的合成全局片段**，局部 ROI 临时为 **32×32、chunk=4**，不是候选训练配置的 128×128/chunk=8。测得峰值约 **11.94 GiB**，不能外推为 81 帧正式分辨率的显存或速度。双卡检查只包含手部模块，不包含完整 Wan DDP。两次更新只能检验执行与梯度，不能判断收敛；K-free 第二次 loss 上升也不能据此判断长期趋势。

双卡检查存在既有 1×1 RayHead 卷积梯度 stride 与 DDP bucket stride 不一致的性能提示，未影响有限值、跨卡梯度一致性与恢复；正式吞吐验收仍需关注。诊断进程完成后已退出，没有留下训练进程。

### 正式训练前仍需完成

1. 在独立目录构建完整 train/val 困难窗口索引，冻结源码及配置；测试清单只能原样复制，不读取测试几何设计策略。B0 也使用同一验证索引。
2. 真实片段短训与过拟合检查，正式 81 帧/全分辨率/8 GPU 的显存和吞吐检查。
3. B0、单项与候选组合的等曝光消融；仅用完整验证选择配置，最后执行独立完整测试。

本次没有启动上述正式实验，没有新的可发布 Fusion 训练权重，没有重新评测已发布 Core 权重。实现范围和命令见 [当前 Fusion 指南](FUSION_R4.md)。

## Core 发布记录（2026-09-10）

日期：2026-09-10。此记录区分源码回归、权重完整性和完整模型实验，不将其中一种替代另一种。

### 源码与权重

- 发布前开发工作区已有 122 项 CPU 回归通过，其中包含真实 MANO 资产检查和两份保留源码的 Core FP32 对齐。
- 公开仓库不分发私人部署记录和历史源码归档。对应的两项归档对齐测试在缺少可选 fixture 时显式跳过；MANO 测试也只在用户提供合法资产后执行。
- 新增 tensor-only 权重加载、哈希、架构/solver/推理配置检查、CLI 标志互斥与公开权重清单一致性测试。
- 两份原最终 Core checkpoint 均先核对固定 SHA-256，以受限 `weights_only=True` 解析并导出。每份导出 755 个 tensor，逐项确认 dtype、键和数值完全相等。
- 公开推理/评测的 `--released-weights` 路径只读取 safetensors，不执行 pickle、不恢复优化器，不允许将 Core 权重加载为 Fusion。

发布目录执行完整 CPU 回归：**135 passed, 2 skipped, 12 warnings in 26.76s**。两项跳过均为未公开的历史源码 fixture；真实 MANO 检查已运行。12 项 warning 来自既有 MANO / NumPy / SciPy 依赖弃用提示。静态检查同时通过：62 个 Python 文件 AST、全部配置/清单 JSON、Ruff 致命错误检查。

两份实际 `.safetensors` 均经公开加载器通过 SHA-256 和元数据校验，确认全部 755 个 tensor 为有限 FP32 值、合计 175,503,289 参数；每份另外将 150 个手部 tensor 严格载入带真实 MANO 的 Core 手部模块，键、形状和数值匹配。元数据不包含私人服务器地址、用户目录或原训练配置路径。

### 该次发布未验证内容

本次发布没有启动新训练，也没有重新跑完整 Wan 的 GPU 推理或 728 段测试。原训练文件保持不变；历史测试结果不作为当前运行层的重新评测结果。没有据此宣称模型精度改善、实时运行或安全关键用途可用。
