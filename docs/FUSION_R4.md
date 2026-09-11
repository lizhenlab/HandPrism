# HandPrism-Fusion r4：实现与实验指南

此版本把评测、数据契约和六项候选改进一次性接入训练、验证、评测与片段推理。它不是已训练模型，也不预设所有模块同时启用一定最好。Core 的结构、默认配置和已发布权重不变。

训练、验证、测试只允许 ARCTIC/HOT3D。数据集白名单、路径检查、冻结清单、架构与 checkpoint 身份检查继续生效；不引入其他手部数据、外部伪标签或手部网络初始化。

## r4 训练前修订

本轮保留六项候选模块，修订训练稳定性与验证覆盖，不扩大主干：

- 给已有深度头增加独立 `log_depth` L1 监督，目标为有效 MANO 腕点的相机 Z 的自然对数（Z > 0.01 m），默认权重 0.5。不是腕点到相机的欧氏距离，也不把 GT 输入 K-free 前向。有有效正 Z 标注时，即便几何求解全部回退到独立位置先验，深度头仍能获得梯度。启用腕点先验时禁止关闭该监督；B0 保持原目标，可单独消融 `depth-loss`。0.5 是待验证的工程起点，不宣称最优。
- 在旋转乘法、平方、归一化、BCE 等非线性计算之前，为无效 GT 选择有限占位值；有效 GT 非有限时直接报错，模型预测非有限也直接报错。不会用全局 NaN 替换来掩盖模型错误；标注源张量不改写。
- ROI 教师过渡按 clip/hand 选择完整 GT 框或完整预测框，概率 `max(0,1-step/1000)`；选择和归一化扰动在整个片段内一致，不再线性混合远离的框中心。GT 不足三点时该帧使用预测框，验证/评测/推理仍禁止 GT 框。
- 使用索引版本 2：在同一批 held-out recording 内扩展固定、互不重叠的验证窗口；快速验证与完整验证使用不同且可核验的集合。原 Core 清单、train/val/test 身份隔离及 test 的字节内容不变。

`train.jsonl` 新增 `fusion_diagnostics`：几何求解比例、平均几何/先验融合权重、ROI 有效比例/GT 点覆盖率/监督点数，以及累积后裁剪前的分支梯度范数。首步零初始化残差内部梯度为零是正常的；诊断不是精度或已校准置信度。

## 模块与开关

| 模块 | 实现 | 主要开关 |
| --- | --- | --- |
| M1 最终关节 readout | 最后一次时空/FFN 更新后，joint query 再读取空间 memory；手部 query 的基础路径保留 | `fusion.final_readout` |
| M2 原始 RGB 局部细节 | 同帧原始 RGB → 预测 ROI → 小 CNN → 姿态、anchor、direct joint 的零初始化残差 | `fusion.local_rgb` |
| M3 关节引导 MANO | 按 MANO 原生关节顺序组合 joint/hand 特征，输出 6D 姿态残差；另有直接 MANO root GT loss | `fusion.joint_mano`、`loss_weights.joints_root_mano`、`direct_mano_consistency` |
| M4 独立位置先验 | 时序 hand 特征预测相机空间 XYZ 和 Laplace log-scale；保留深度头独立 Z 监督 | `fusion.temporal_wrist`、`loss_weights.wrist_prior/log_depth` |
| M5 质量加权几何 | 监督 joint 质量、加权闭式 XY、0–3 次像素域 Huber 重加权；保留数值检查和回退 | `fusion.reliability`、`geometry_refinement.robust_iterations` |
| M6 运动与训练策略 | FP64 时间差下的速度/加速度误差，片段一致外观增强，保留均匀采样的困难窗口混合 | motion loss 权重、`augmentation`、`hard_window_fraction` |

边缘策略单独控制：`decoder.anchor_offset_cells` 为 0/0.5/1 cell；`fusion.edge_quality` 在最外一个 feature cell 内平滑降低观测权重，原硬几何有效性判断仍保留。有限 offset 不等于能准确预测任意出画位置。

`configs/handprism_fusion_{standard,kfree}.json` 是全功能候选；`handprism_fusion_b0_{standard,kfree}.json` 关闭全部新增模块/训练策略。B0 保留基础全局计算和旧 loss，但共享 r4 的时间/像素契约、固定验证划分与新指标，不应当成旧训练分数的无条件替代。

### 重要约束

- 局部细节从原始图像读取，最长边默认限制为 1408，不先缩至全局图再放大。全局输入仍为 ARCTIC 480×672、HOT3D upright 480×480。
- ROI 默认 128×128、分块 8 只手处理。前 1000 optimizer step 可使用带扰动的 GT ROI，逐步过渡到自身预测；验证、评测、推理禁止 GT ROI。失败 ROI 的局部残差为零，不删除全局预测。
- ROI 使用全图归一化坐标和半像素中心映射，不改变全局标定或鱼眼 ray；ARCTIC 的 Fusion resize 内参遵守相同像素中心约定。Core 保留原约定。
- 姿态残差从零开始。MANO GT loss 可与 joint-MANO 模块分开对照；一致性 loss 默认前 1000 step 为零，随后 1000 step 缓慢增加，不替代两支各自的 GT 监督。
- 质量头目标为 `exp(-anchor_error_px/8)`，出画或人工遮挡为零；未知真实遮挡不伪造成负例。用于几何的质量权重 detach 且设置下限，不能靠把权重全降到零逃避姿态 loss。
- 独立腕点使用 Laplace NLL 监督。融合中的几何尺度是工程近似，不是已校准的真实测量协方差；最终必须看验证集上的覆盖率、风险和错误高置信比例。
- `geometry_refinement.depth_refine_fraction` 提供有先验、有限区间的深度微调（最多 ±25%），默认 **0，未启用**。不得当成自由焦距/自由深度 PnP。
- `pnp_rms_pixels`、`pnp_solved` 描述几何候选；若启用独立先验，它们不等于混合后的最终输出残差。另读 `geometry_weight` 和最终 MANO reprojection EPE。
- K-free 模型前向仍不读取 GT 标定。评分/loss 使用 GT 相机不等于将标定输入模型。

## 时间、有效性与数据索引

`HandPrismSample` 增加 FP64 `timestamps` 和来源、`in_frame`、`observed/observed_valid`、`visibility_valid`、`synthetic_occluded`、可选 uint8 `rgb_high`。

- ARCTIC 采用经核对的 release 帧号 / 30 Hz，不捏造逐帧设备时间戳；图像索引仍核对 `ioi_offset`。
- HOT3D 使用 RGB TIME_CODE 纳秒转换为秒，并核实窗口中每个 mask 时间确实属于 RGB 流，避免 `CLOSEST` 静默重复帧。
- 时间戳必须有限、非递减。重复时间、超过 0.15 秒的间隔及无效几何会切断运动监督，不把时序缺口当高速运动。
- `valid_*` 是标注能力，`in_frame` 是投影范围，`observed` 是有证据的视觉可见性。目前适配器没有可靠逐关节遮挡 GT，因此 `observed_valid=false`；不能声称已获得精确遮挡标签。Fusion visibility head 当前监督的是几何在画面内存在性。
- root-relative 手部运动与绝对腕点轨迹分别计算，单位为 m/s、m/s²。旧 `acceleration` 是自身平滑正则，完整候选将其置零；不得和真实动作误差混称。

新索引构建见 [运行说明](RUNNING.md#数据准备)。构建只解码 train/val 几何，原样复制 test 清单；保留 recording 和 subject/participant 隔离。候选训练窗口必须落在原 `valid_ranges` 或 `start_min/start_max` 内。默认 35% 概率从有标签的困难窗口抽样，其余保持原均匀窗口路径；两个数据集仍按 7:9 选择。

困难阈值在测试前固定：边缘 32 px、小手框对角线 48 px、腕点速度 1 m/s、出画及有可靠标签的遮挡。不依据测试表现调这些阈值。颜色、模糊、JPEG、降分辨率再恢复及人工遮挡采用片段一致参数；不做改变视场而忘记同步内参的几何缩放/翻转。

## 消融与曝光预算

以下命令只生成新配置，不启动训练，不覆盖既有配置。`--stage M1` 表示 **B0 + M1**，不是隐式累加；`all` 才启用全部候选。先比较 B0/单项，再选择组合。

```bash
bash scripts/v3_python.sh scripts/make_v3_ablation.py \
  --architecture handprism-fusion --base configs/handprism_fusion_standard.json \
  --stage M1 --clip-budget 128000 --world-size 8 \
  --output configs/ablation_standard_m1.json

bash scripts/v3_python.sh scripts/make_v3_ablation.py \
  --architecture handprism-fusion --base configs/handprism_fusion_standard.json \
  --features readout mano-loss --clip-budget 128000 \
  --output configs/ablation_standard_readout_manoloss.json
```

细粒度 features：`edge`、`readout`、`local-rgb`、`joint-mano`、`mano-loss`、`consistency`、`wrist-prior`、`depth-loss`、`quality`、`irls`、`motion`、`augmentation`、`hard-sampling`、`loss-rebalance`。`consistency` 必须同时保留 `mano-loss`；`wrist-prior` 必须同时保留 `depth-loss`；质量/腕点头必须各自带监督 loss。

128000 clips 在 8 卡默认配置下，Standard 为 2000 step，K-free 为 4000 step。测试前固定曝光预算、batch、初始化、样本流和验证集合。可选模块不消耗共享全局/ray 权重的初始化随机流；GT ROI 的训练随机扰动仍属显式数据路径差异。改变 world size 必须重新生成/冻结预算，不能沿用原曝光声明。

`train.jsonl` 记录 `global_clips_seen/global_frames_seen`、各 loss 与 `loss_audit.weighted`。加权和会核对 total；它不是梯度大小，Laplace NLL 也可能为负。诊断工具的 `--gradient-audit` 可另外测量每项 loss 对 decoder 的梯度范数，不能把 loss 数值直接当作梯度贡献。

短程配置用 `scripts/train.py` 启动；完整 `run_pipeline.py` 固定两种 solver 各 20000 step，不用于短程消融。短程结果只筛选方向，不证明已收敛。

## 验证与 checkpoint 选择

指标协议为 2；详见 [EVALUATION](EVALUATION.md)。两种 solver 同时报告两个 EPE、root/camera-space 关节及网格误差、OOS 绝对位置、覆盖率、动态误差、困难分层、质量与先验校准诊断。旧指标保留原定义，禁止混用同名旧分数。

索引构建在原 val recording 的连续有效区间内，以 81 帧间隔枚举互不重叠窗口，每个 recording 最多保留 12 个：一半预算均匀覆盖时间，其余按困难分层补足，短 recording 不复制窗口。ARCTIC 遵守 `usable_start/usable_stop`；旧 HOT3D val 没有保存完整区间时，重新读取同一组 masks 并核对原统计，绝不根据 `num_frames` 猜测可用区间。最终数量和分层计数写入 `split_report.json`，readiness 核查窗口、摘要和完整验证确实大于快速验证。

默认每 500 step 运行固定分层快速验证：**每数据集全局共 16 个片段**，由 `validation_clips_per_dataset` 指定，不随 GPU 数量倍增；不同 rank 无重复分片，同一分层内轮换 recording。每 2000 step 及配置的最终 step 运行完整冻结 val，日志记录实际窗口数。**只有完整验证** 可以选择 best。固定规则对两数据集等权平均：

`CameraMPJPE_mm/100 + MPJPE+OOS_mm/50 + AnchorEPE_px/20 + 5×(1−ExistenceCoverage) + 5×(1−F1)`

这些尺度是固定工程选择，不是外部榜单分数，也未在 test 上拟合。任一数据集所需指标缺失/非有限时拒绝选 best。每次实际保存 checkpoint 后，`best.json` 才绑定该文件、step、score 与协议；`latest.json` 和 final checkpoint 继续保留。完整测试仍固定 final step 20000；如要报告 validation-best，须在测试前另行明确选择策略并单列成绩。

更多窗口改善时间/难例覆盖，但没有增加独立参与者数量；不能据此宣称跨参与者泛化已经充分验证。新验证分数不与旧单窗口验证分数直接比较。测试集不参与窗口、权重、阈值或模块选择。

## 有界运行检查

需要合法 MANO 和对应环境。输出必须是新文件；不加载已训练的 Core 权重。

```bash
# ARCTIC/HOT3D 各五帧训练样本，检查原始 RGB 与时间契约
bash scripts/v3_python.sh scripts/validate_fusion_runtime.py \
  --architecture handprism-fusion --config configs/handprism_fusion_standard.json \
  --mode real-data --source-manifests data/manifests/two_dataset_v2_clean \
  --output runs/r4_checks/real_data.json

# 真实 MANO + 合成特征，两次 CUDA/BF16 更新、恢复与 loss 梯度检查
bash scripts/v3_python.sh scripts/validate_fusion_runtime.py \
  --architecture handprism-fusion --config configs/handprism_fusion_standard.json \
  --mode heads --gradient-audit --output runs/r4_checks/heads.json

# 可用 -m torch.distributed.run 包装 --mode heads --accumulation 2
# 多卡模式不要启用 --gradient-audit
# --mode wan 另包含实际 VAE/DiT，仍只有 5×64×64 合成片段和两次更新
```

这些检查不是正式训练，不测量真实精度，也不等于 81 帧全分辨率多卡训练已验收。完整困难索引构建、真实片段短训、正式分辨率多卡显存/吞吐、多个随机种子和完整测试仍须在冻结实验后执行。实现接口一次备齐，不代表可以省略这些验收。

有界工具另外核查第二次更新的非零分支梯度，包括深度头、局部 CNN、最终 readout、MANO/腕点/质量头；Wan 模式还要求 patch embedding 及第 0、15 层 LoRA 收到梯度。后续 block 不在前向 feature tap 内，不把其无梯度误报为训练故障。

完整流程要求当前工作区有自身的已提交、干净 Git 状态；不能用父目录的干净状态替代。隔离源码快照允许诊断和显式开发试验，以源码/config 哈希记录身份，不伪装成父仓库提交。正式实验前仍须冻结实际源码。

## 兼容性与范围

实现 ID 为 `handprism-fusion-r4`，训练序列化契约版本仍为 9；架构实现 ID 和完整配置均进入 checkpoint 身份检查。r3 的训练目标/ROI 课程/验证索引不同，不允许以 r4 自动续训；新实验从 step 0 开始。Core 的实现 ID、公开默认配置、发布推理包与权重清单未修改。

本轮没有将局部 CNN 混合精度、数据读取缓存或更大 batch 作为默认改动：这些属于须先测量峰值显存和真实吞吐的性能实验，不以未验证的提速换取数值/数据契约变化。全分辨率 81 帧、真实数据、多卡短训与消融仍是正式长训前的验收门槛。

暂未加入更大主干、全量解冻、SLAM、接触物理、跨窗口记忆、caption/context 改动、外部伪标签或双目分支。MP4 封装和左主右辅属于后续输入/推理工作，不通过改模型名字暗中启用。
