# 参考资料与第三方组件

这些资料说明研究背景与组件来源，不表示相关作者或机构认可本项目，也不将已有研究成果或第三方组件声明为本项目原创。具体设计与验证范围以本仓库为准。

- [ACE-Ego-Hand](https://arxiv.org/html/2608.20308v2)：视频手部建模、有效相机拟合与几何求解的方法参考。
- [Wan2.2-Fun-5B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control)：预训练视频主干；[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)：对应模型实现。
- [MANO](https://mano.is.tue.mpg.de/) 与 [SMPL-X](https://github.com/vchoutas/smplx)：参数化手部几何与加载组件，模型资产需单独取得许可。
- [ARCTIC](https://github.com/zc-alexfan/arctic)：Zicong Fan 等，*ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation*，CVPR 2023。
- [HOT3D](https://github.com/facebookresearch/hot3d)：*HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos*，CVPR 2025。
- [HOT3D 官方 MANO 实现](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/mano_layer.py)：标签解码和左手 MANO `shapedirs` 条件修正的核对来源。

HandPrism 自有代码和文档采用 [MIT](../LICENSE)。第三方素材和权重不因此改变许可；详见 [第三方声明](../NOTICE.md) 与 [权重使用说明](../WEIGHTS_TERMS.md)。
