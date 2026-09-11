# 模型选择与输入

## 架构与相机模式

| 项目 | HandPrism-Core | HandPrism-Fusion |
| --- | --- | --- |
| 架构标识 | `handprism-core` | `handprism-fusion` |
| 实现 ID | `handprism-core-r1` | `handprism-fusion-r4` |
| Query 交互 | 各 query 分别在时间轴交互 | 时间 × query 联合交互 |
| Direct joints | 插值隐特征后回归坐标 | 先回归坐标，再插值坐标 |
| PnP 残差 | bearing × 图像对角线代理 | 相机像素投影或预测 ray 逆映射 |
| K-free 接受条件 | 方差、有限值、焦距区间 | 另加正向 ray 和 RMS 兼容性检查 |
| 可选精修 | 无 | 最终关节 readout、逐帧 RGB ROI、关节 MANO、腕点先验、质量加权 |
| 完整训练权重 | Standard / K-free | 尚无 |

Standard / K-free 是相机模式，不是架构。每个生产入口必须显式填写 `--architecture`；配置、模型实例和权重身份必须一致。两种架构可能参数形状相同，不能因此混用权重。

配置在 `configs/handprism_{core,fusion}_{standard,kfree}.json`；Fusion 另提供 `handprism_fusion_b0_{standard,kfree}.json` 基础对照。新训练 checkpoint 使用 `format=handprism-training-checkpoint`、契约版本 9，包含架构、实现 ID 和完整配置；恢复还检查源码与数据契约。Fusion r2 checkpoint 不能作为 r3 训练恢复文件。发布的 Core 推理包仍由原固定哈希和身份规则读取。

## 推理输入

`scripts/infer.py` 使用 NPZ 片段：

- `video`：`uint8 [T,H,W,3]` RGB；T 为 1、5、…、81，H/W 为正的 32 倍数。
- Fusion 开启 `local_rgb` 时必需 `rgb_high`：`uint8 [T,H_high,W_high,3]`，与 `video` 同帧、同视场、相同比例，分辨率严格高于全局图。入口能检查形状，不能自动证明图像来源，调用方须保证原始细节与对齐。
- 可选 `timestamps [T]`：有限、非递减的秒数，以 FP64 保存。运动训练必须有可靠时间轴；推理不从帧数猜 FPS。
- Standard 必需 `intrinsics [3,3]`，对应当前输入尺寸；可选 `distortion [8]`。
- 原生鱼眼需要 `camera_model=fisheye624_upright`、`camera_parameters [15]`、`source_image_size [2]` 和 `calibration_ray_field [H/32,W/32,3]`，遵循本项目相机适配器的坐标约定。
- K-free 不读取输入中的 GT 相机字段。

入口不自行缩放、猜标定、处理整个 MP4 或拆分双目。Fusion 不表示具有新的双目模型分支。

输出 `prediction.npz` 包含左右手参数、关节、腕点、MANO 原生平移、2D anchors、存在性/可见性和几何诊断；`provenance.json` 记录架构、solver、输入和权重哈希。MANO 网格可以通过已合法取得的模型资产计算，当前 NPZ 不直接保存顶点数组。

启用对应模块时额外导出 `wrist_prior`、`wrist_log_scale`、`reliability_logits`、`local_roi_valid/bounds`、`anchor_quality`、`geometry_weight`、`joint_weights` 及时间轴。它们是诊断输出，不是已经校准的可靠性保证；含义见 [Fusion 指南](FUSION_R4.md)。

## 权重入口

- GitHub 发布的 `.safetensors`：使用 `--released-weights`。加载器先检查固定 SHA-256、Core 身份、solver、步数、数据与推理设置；再严格检查所有 trainable tensor 的键和形状。不执行 pickle，不恢复优化器。
- 本机新训练的可信 checkpoint：不添加上述标志，架构与完整配置必须一致。不要将不可信 `.pt` 文件交给训练 checkpoint 加载器。
- 兼容接口 `--legacy-weights` 仅接受两份固定哈希的原最终 Core checkpoint，不能与 `--released-weights` 同用。公开分发首选 tensor-only 格式。

Standard 权重必须配 Standard 配置，K-free 权重必须配 K-free 配置。发布权重允许更改数据/资产位置，不允许改变推理数值设置；不支持作为训练恢复文件。

## 验证范围

发布文件与原 checkpoint 的 trainable tensor 逐项相等，不进行量化或降精度。当前运行层与原训练存在明确边界：FP32 几何、修正后的标签和指标可能改变输出或评分，不能沿用历史结果作为当前测试指标。CPU 检查不替代完整主干 GPU 验收。详见 [模型卡](MODEL_CARD.md) 与 [验证记录](VALIDATION.md)。
