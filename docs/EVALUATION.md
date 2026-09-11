# HandPrism 评测口径

本页对应 `src/handprism/evaluator.py` 与 `src/handprism/metrics.py` 的当前实现。比较结果时须同时核对架构、相机模式、清单哈希、checkpoint、指标定义与数值单位；不能只根据指标缩写认为不同实验可直接横比。

## 输入与检测匹配

完整测试使用冻结的 728 个 81 帧片段：ARCTIC 291，HOT3D 437，分别汇总数据集指标。模型固定输出左、右两个槽，按同侧匹配，不做跨手匹配。

存在性概率大于 0.5 且至少一个关节位于图内、深度大于 0.01 m 的预测是候选。真实手也按有效关节投影判定是否在画面内。候选 MANO 投影框与真实框的 IoU 大于 0 才算匹配；真实框宽高各扩大 10%。漏检计 FN，未匹配的候选计 FP。`FAcc` 是同一帧左右两手均无 FP/FN 的帧占比，不是姿态误差达标率。

Standard 与 K-free 评测均使用真实标定投影以计算统一的检测匹配和误差。共享批处理接口保留标定字段，但模型的 K-free 分支忽略真实内参、畸变和相机参数，并且评测调用不向该分支传真实 ray 场；这些标定只参与监督或评分。用户 RGB 推理入口还会直接丢弃 K-free 输入中的标定字段。预测的 visibility logit 不直接决定这里的在画面内分组。

## 数值指标

| 指标 | 当前计算方式 |
| --- | --- |
| `MPJPE-p_mm` / `PA-p_mm` | 在画面内真实手的腕点对齐 / 相似变换对齐关节误差；匹配时使用预测，漏检时使用 canonical 手形误差作为惩罚 |
| `EPE2D-p_px` | 有效二维关节平均像素误差；Standard 使用预测 anchors，K-free 使用预测三维手经真实相机投影后的坐标；漏检惩罚为图像对角线长度 |
| `GO-p_deg` | 全局旋转的测地角误差；漏检时以单位旋转作为预测 |
| `CT-p_m` | MANO 原生平移误差；漏检时以零平移作为预测 |
| `Wrist-p_m` | 腕点相机空间位置误差；漏检时以原点作为预测 |
| `MPJPE-matched_mm` | 仅匹配手的腕点对齐关节误差 |
| `MPJPE-IV_mm` / `MPJPE-OOS_mm` / `MPJPE+OOS_mm` | 所有有效真实手按在画面内 / 出画 / 合计分组；不依赖检测是否匹配，也不使用漏检惩罚 |
| `Jitter_mm_per_frame2` | 在同一片段、同一只手的连续匹配区间内，三维关节二阶帧差范数的均值；区间至少 3 帧 |

后缀 `-p` 表示包含上述漏检惩罚，不能当成仅匹配样本的平均误差。无对应样本的指标输出 `null`，不是零误差。Jitter 没有按时间戳或帧率换算，不是 mm/s²；不同帧率之间不应直接比较。

Standard 与 K-free 的 `EPE2D-p_px` 预测来源不同，比较时必须注明模式。当前 `CT-p_m` 与历史旧 CT 也不是同一定义。腕点和 MANO 原生平移在形状相关的 J0 偏移下不同，导出时分别使用 `wrist_camera` 和 `mano_translation`。

## 指标协议 2 的统一增量

上述历史指标继续保留原定义。新报告顶层和各数据集均标注 `metric_protocol_version=2`，两种 solver 统一增加以下指标，不依赖预测是否匹配来隐藏误差：

| 指标 | 含义 |
| --- | --- |
| `AnchorEPE_px` | 有效 GT 二维关节上的预测 anchor 像素误差，两种 solver 同一来源 |
| `MANOReprojectionEPE_px` | 有效 MANO/二维 GT 上的最终三维关节经真实相机投影的像素误差；非正有效深度用图像对角线惩罚 |
| `CameraMPJPE_mm`、`WristAbsolute_mm`、`DepthAbsolute_mm` | 无对齐的相机空间关节、腕点三维和腕点 Z 误差 |
| `RootPVE_mm`、`CameraPVE_mm` | 根节点相对和绝对位置下的 MANO 顶点误差 |
| `Root/WristVelocityError_m_s`、`Root/WristAccelerationError_m_s2` | 预测运动相对 GT 的速度/加速度误差，分别报告手部相对形变与腕点轨迹 |
| `ExistenceCoverage` | 有效 GT 手上存在性分数 >0.5 的比例；需同时看 F1/误检率，不是几何匹配召回率 |

新增指标附各自有效数量。无标签或无有效时间间隔时为 `null`，不是零误差。时间差先在 FP64 中计算；重复时间、间隔 >0.15 秒及无效关节会断开速度/加速度区间。Core 没有时间轴时动态增量不可用，不能据此填零。

按 GT 固定分层输出 `RootMPJPE_mm / CameraMPJPE_mm / WristAbsolute_mm / DepthAbsolute_mm / ExistenceCoverage` 和手帧数：边缘（32 px）、小手（框对角线 <48 px）、快动作（腕速 >1 m/s）、OOS、可信遮挡。遮挡标签未知时该分层没有样本，不用投影在图内代替真实遮挡标签。

`oos_within_clip_le0p5s / oos_within_clip_0p5_1s / oos_within_clip_gt1s` 按当前片段内连续 OOS 的首末帧时间差分层。窗口边界、时间缺口、邻近无效标注造成截断，另报 `OOS_censored_runs`。这是片段内观测跨度，不是完整出画时长，也不代表跨窗口关联。

启用新质量/先验头时，另报 `QualityTargetMAE`、五箱 `QualitySoftECE`、固定阈值 0.25/0.5/0.75/0.9 下的 anchor 风险/覆盖率；质量目标是软定位精度 `exp(-error_px/8)`，不是二元遮挡概率。腕点报 Laplace NLL（省略常数）、平均尺度、95% **逐坐标轴** 区间覆盖率，以及尺度向量范数 <2 cm 却腕点误差 >10 cm 的比例/计数。空高置信集合输出 null，不能当成无失败。

模型选择只用完整 val 的固定精度/覆盖复合分数，定义见 [Fusion 指南](FUSION_R4.md#验证与-checkpoint-选择)。最终 test 不参与阈值、loss、采样或 best 选择。

## 求解诊断与完成条件

求解器的 `pnp_rms_pixels` 是兼容字段名，解释时必须读取 `pnp_residual_kind`：Core 为 `bearing_diagonal_proxy`，不是实测像素误差；Fusion 为 `native_pixels`，对应原生相机投影或预测 ray 场逆映射的像素残差。两者不能按字段名直接比较。

有先验融合时，上述 RMS 是融合前几何候选的诊断；最终质量看最终输出的 reprojection/absolute 指标，结合 `geometry_weight`，不能把候选接受率等同最终准确率。

完整流程只接受包含两数据集完整计数、step 20000、匹配的架构/实现 ID/solver/checkpoint 哈希且数值有限的 metrics。subset、诊断输出或仅有进度文件不代表完成。CPU 回归仅验证程序约定，完整模型精度和吞吐仍需实际 GPU 实验验收。
