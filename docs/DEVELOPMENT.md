# HandPrism 开发与设计说明

当前有 HandPrism-Core 与 HandPrism-Fusion 两种显式架构，详见[模型选择](MODELS.md)。输入为第一人称 RGB 视频片段，输出为相机坐标系下的左右手几何与运动参数；不将相机空间轨迹描述为已验证的世界坐标轨迹，也不承诺实时推理性能。

## Fusion 的时序与几何路径

1. 使用 Wan 视频特征。保留预训练 VAE 的归一化与视频主干整体 32 倍的空间步幅，不修改 patch kernel 来改变网格。
2. Decoder 在 `[T × Q]` 查询流上进行双向时间交互，RoPE 只编码帧时间；寄存器、关节与手部 query 可以交换信息，不写回视觉 memory。
3. Direct-joint head 在 latent 时刻预测三维坐标，再插值到 RGB 帧；不对已插值特征做额外非线性坐标回归。
4. MANO 提供参数化手形，ray 场和 Mixed-PnP 用于相机空间定位。投影阈值使用真实像素单位，并记录失败和回退状态。

Core 使用相同的视觉主干接口，但各 query 独立沿时间交互，direct-joint 在插值后的隐特征上回归坐标；PnP 保留 bearing 误差乘图像对角线的历史代理，不能标作真实像素误差。Core 不启用边缘 residual head。上述差异由架构分支实际执行，不只是命名不同。

## Fusion 的 Standard 与 K-free

- **Standard**：使用已知的畸变针孔或原生 Fisheye624 相机投影。
- **K-free**：从预测 ray 场拟合有效针孔相机；兼容性不足时走直接射线分支，需要像素残差时数值反解预测 ray 场。推理不输入 GT 相机内参。
- **兼容性门限**：有效针孔拟合需满足归一化二维 RMS ≤ 0.01。该门限是配置约定；畸变针孔也可能走直接射线分支，不按数据集名称决定分支。
- **显式失败**：射线不可逆、投影越界或迭代未收敛时保守回退，不伪造有效投影。记录投票数、像素 RMS、投影有效性和失败位码。

默认 `L_fit` 只监督有有效 ray 标签、且目标符合针孔近似的样本。监督目标分支仅用于 loss；预测 residual 不能直接关闭自己的 `L_fit`。相关方法背景见 [参考资料](REFERENCES.md)。

Core 也支持两种 solver，但其 K-free 拟合只应用方差、有限参数与焦距区间检查，不采用 Fusion 的正向 ray / RMS 门限；`L_fit` 固定为 `core_bearings`。Core 的归一化拟合 RMS 只作诊断，不能把两者的相机接受率或 PnP 代理残差直接当成同口径指标。

## 两种架构共享的运行层修正

- **标签一致性**：HOT3D 原始 PCA MANO 层与 canonical 层统一使用幂等左手 shapedirs 修正。
- **FP32 几何区**：空间热图累加、小 readout、ray head、旋转、MANO、求解与损失关闭外部 autocast；Wan 和主要注意力继续 AMP，几何导出保持 FP32。
- **按片段归一化**：先计算每片段有效项均值再平均，空片段贡献零。等大小 microbatch/rank 重新分组不改变片段权重；默认仍为 batch=1、accumulation=8/4。
- **位置与指标**：分开导出 `mano_translation` 与 `wrist_camera`。`CT-p_m` 使用 MANO 原生 translation，`Wrist-p_m` 使用腕点误差；训练中的 `translation` loss 仍是腕点监督。v2 旧 CT 不与新约定直接横比。
- **质量诊断**：分别记录 direct/MANO root-relative MPJPE、anchor EPE、图像最外侧 5% 的边缘 EPE/数量、腕点/深度误差、检测与姿态指标、回退原因和 anchor 聚集情况。K-free 额外区分数值有效、目标兼容与实际监督比例。
- **有限缓存**：HOT3D 标定 ray 场使用最多 32 项 CPU 张量缓存。返回副本，不缓存或跨进程共享 VRS reader，不缓存训练中的 Wan/LoRA 特征。
- **运行保护**：训练、评测和 supervisor 输出只能位于当前隔离工作区的 `runs/`；拒绝历史 v2 路径及软链接越界。运行契约记录实际 `src/`、`scripts/` 内容哈希。

诊断警告不等于自动停止条件。新增代码和 CPU 测试不证明完整模型精度或吞吐已改善。

## Fusion 的可选实验开关

以下功能已实现，但默认不启用替代策略。使用固定训练/验证片段先做单变量对照，测试集不参与选择。

### 边缘 anchor 修正

`decoder.anchor_offset_cells` 默认为 0；可选 0.5/1.0 cell 的零初始化、有界 residual head，允许 anchor 超出网格中心的凸包。

该 head 只扩展定位表达范围，不解决 K-free 在边界处的 ray 逆映射歧义，仍须保留求解器有效性检查。它是否改善边缘手形和出画误检，须经验证。

### 相机拟合监督

`kfree_camera_fit.target` 有三种显式配置：

- `pinhole_compatible`：默认策略，保留目标兼容性筛选。
- `effective_camera`：以真实 ray 场拟合出的有效相机作为监督目标。
- `raw_bearings`：只按数值有效性筛选，使用原始 bearing 目标。

后两种策略不改变推理 RMS guard，也不向推理分支输入 GT 相机。按实际训练网格的诊断，ARCTIC 的 236 个训练相机均通过 0.01 门限，HOT3D 的 92 个均未通过；因此默认关闭 HOT3D 的 `L_fit`，但保留其他监督。早期 ARCTIC 0.01021 诊断使用另一种网格，不能替代该结果。

下面只生成实验配置，不覆盖原配置、不启动训练；它展示参数用法，不代表正式采用组合方案：

```bash
bash scripts/v3_python.sh scripts/make_v3_ablation.py \
  --architecture handprism-fusion \
  --base configs/handprism_fusion_kfree.json \
  --output configs/ablation_kfree_edge_effective.json \
  --anchor-offset-cells 1.0 --fit-target effective_camera
```

## 已知边界与下一步验证

早期 HOT3D GT-ray oracle 在 126 个 eligible 手帧中接受 111 个，其余因边界 padding 或投影越出可逆网格而回退；Standard 原生投影接受 126/126。这是求解器测试，不是模型准确率。增加 Newton 迭代不能消除边界不可逆性。

更名前的 74 项 CPU 回归覆盖真实 MANO 一致性、CPU BF16 autocast 几何和等大小片段分组的梯度检查；加入两种具名架构后的当前验证范围见[模型选择](MODELS.md#验证范围)。真实 Wan CUDA 前后向、实际多卡同步与恢复、短程学习和可视化仍待验收。

没有执行主干裁剪、全量 VAE 缓存或未经测量的 batch 扩大。优化需先验证前向、梯度和 checkpoint 语义，再测量数据等待、VAE、DiT、decoder、同步、验证和保存耗时。

## 标识与兼容性

项目名与发行包名为 HandPrism / `handprism`，当前 Python 导入名仍为 `dreamhand`。当前机器可读标识为：

- 架构：`handprism-core` / `handprism-fusion`，所有模型入口均须显式选择。
- 实现 ID：`handprism-core-r1` / `handprism-fusion-r2`。
- 运行契约版本：8。
- checkpoint 格式：`handprism-training-checkpoint`，同时记录架构与完整配置。

CLI、配置、模型实例与 checkpoint 架构必须一致；不能依赖同形状参数猜测前向逻辑。两份保留的 Core 最终权重只允许经 SHA-256 校验的显式 legacy 推理/评测入口读取，不接入新优化器状态。新 Core 与 Fusion 实验均需冻结配置并从独立的 step 0 运行开始；新的同契约 checkpoint 可以恢复。原源码、权重与许可保持原状，历史格式标识仅用于归档、兼容识别或拒绝旧格式的回归测试。

评测计数、漏检惩罚与单位见[评测口径](EVALUATION.md)。参数诊断 `scripts/inspect_model.py --architecture ...` 只报告实际实例化的默认 decoder/ray head 参数量，不把主干或 LoRA 的外部估计当作实测值。
