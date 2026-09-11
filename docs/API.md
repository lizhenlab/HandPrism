# HandPrism Python API 与命令入口

项目源码包、导入名和发行包名统一为 `handprism`。项目级类名使用 `HandPrism*`，内部几何与工具函数按职责命名；不保留其他项目命名的导入别名。

## Python 导入

```python
from handprism.config import HandPrismConfig
from handprism.decoder import HandPrismDecoder, HandPrismDecoderOutput
from handprism.model import HandPrismModel, HandPrismOutput
from handprism.system import HandPrismSystem
from handprism.losses import HandPrismLoss, HandPrismPrediction, HandPrismTarget
from handprism.data import HandPrismSample, HandPrismWindowDataset
from handprism.ray import kfree_bearings
from handprism.lora import configure_trainable_backbone
```

源码位于 `src/handprism/`。升级现有工作区后，外部调用方须使用上面的导入名，并重新安装本地包：

```bash
.venv/bin/python -m pip install --no-deps -e .
```

这条命令只更新本地 Python 包安装；首次安装环境仍按 [README](../README.md#安装与推理) 准备依赖。

## 唯一主入口

| 用途 | 脚本 |
| --- | --- |
| 构建数据清单 | `scripts/build_manifests.py` |
| 迁移已冻结清单的数据根路径 | `scripts/relocate_manifests.py` |
| 构建 Fusion 困难窗口索引 | `scripts/prepare_fusion_index.py` |
| 运行前检查 | `scripts/check_readiness.py` |
| 完整训练/评测流水线 | `scripts/run_pipeline.py` |
| 训练 | `scripts/train.py` |
| 评测 | `scripts/evaluate.py` |
| 推理 | `scripts/infer.py` |

架构仍须手动指定 `--architecture handprism-core` 或 `--architecture handprism-fusion`。相机模式为配置中的 `standard` / `kfree`，不由包名或目录名推断。`configs/components.json` 仅提供组件索引，不是训练配置。

主入口直接执行时按脚本的真实位置解析源码与同目录工具模块，不依赖外部 `PYTHONPATH` 或当前目录。推荐仍在仓库根目录使用 `scripts/v3_python.sh`，它会选择该工作区的解释器并设置工作目录；脚本传入的相对配置、数据和输出路径仍按各命令参数约定处理，导入修复不会偷偷改变工作目录。

## 命名与序列化边界

- 命名整理本身不改变参数键、张量形状、网络计算或训练损失；无需仅因命名调整重新训练。独立的 Fusion r4 修订改变了训练目标、ROI 课程与验证索引，须使用 r4 配置从 step 0 开始，不能套用这个命名兼容性结论。
- 已发布 Core safetensors 的字节内容、SHA-256、架构身份和加载校验保持不变。这里只加载 tensor-only 权重，不通过恢复旧 Python 类来加载模型。
- 新生成的清单、运行契约、评测与可视化报告统一采用 `handprism-` 元数据标识。
- 两种已知历史清单格式通过格式标签的 SHA-256 白名单只读识别；原清单不原地改写，文件哈希、样本数量、划分隔离和 ARCTIC/HOT3D 白名单检查不放宽。返回的内存报告使用规范标识，并记录原格式摘要。
- 命名改变会影响源码哈希；训练断点仍遵守配置和源码契约检查，不能据此绕过恢复限制。
- 参考资料继续保留可追溯链接，第三方版权及许可证原文保持不变；项目命名不表示第三方技术均为本项目原创。
