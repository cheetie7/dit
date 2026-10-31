# 阶段 2：ACT 风格的 ResNet-18 视觉编码器

阶段 8 的 Advantage-Weighted Flow Matching 与 10 步窗口配置见 [AWR_FM.md](../awbc_training/AWR_FM.md)。

阶段 3 的状态编码与 Observation Transformer 接口、结构和验收见 [OBSERVATION.md](OBSERVATION.md)。视觉部分继续保持本说明中的接口。

阶段 4 的 Conditional DiT 网络、配置、参数量和测试结果见 [DIT.md](DIT.md)。

阶段 5 的 Conditional Flow Matching Objective、逐样本 loss、padding 和权重接口见 [FLOW_MATCHING.md](FLOW_MATCHING.md)。

阶段 6 的完整 Policy、Euler 推理、原始观测预处理和反归一化接口见 [POLICY.md](POLICY.md)。

阶段 7 的训练、EMA、固定条件验证、普通 checkpoint 和恢复见 [../awbc_training/README.md](../awbc_training/README.md)。

阶段 8 已将新实验默认 horizon 调整为 10，并加入 Advantage Weighting；接口与来源约定见 [../awbc_training/AWR_FM.md](../awbc_training/AWR_FM.md)。此前阶段的 16 步实测记录保持原样。

`vision.py` 仅实现视觉部分：ImageNet ResNet-18 → layer4 → 1×1 投影 → 独立的内容与位置 tokens。

参考 [ACT backbone](https://github.com/tonyzhaozh/act/blob/main/detr/models/backbone.py)、[投影与视觉输入](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)、[2D 位置编码](https://github.com/tonyzhaozh/act/blob/main/detr/models/position_encoding.py)。按本项目完整要求，多相机采用逐相机序列拼接，并增加可学习的 Camera Embedding。

## 接口

```python
from awbc_models import ACTVisualEncoder, VisualEncoderConfig

encoder = ACTVisualEncoder(VisualEncoderConfig(
    d_model=256,
    pretrained=True,
    backbone_mode="frozen",
    num_cameras=len(train.source.cameras),
))
images = batch["observation"]["images"]  # [B,N,3,H,W]，已归一化
output = encoder(images)
# 或 encoder.encode_cameras(images)
visual_tokens = output["visual_tokens"]
visual_pos = output["visual_pos"]

# 单张 RGB 图像仍保留 batch 维：
single = encoder(images[:1, 0], camera_index=0)
# 诊断用中间 feature maps：
maps = encoder.forward_features(images[:, 0])
```

公开 forward 严格返回两个字段：`visual_tokens` 和 `visual_pos`。中间张量通过 `forward_features` 单独获取。相机顺序必须在训练与推理中保持一致，索引对应 `camera_embedding` 的行；实际相机数量通过 `num_cameras` 配置，验收脚本直接从 Dataset 读取。

## 特征形状

默认 H=240、W=320、hidden_dim=256 时：

| 阶段 | 单相机 | 三相机最终输出 |
|---|---|---|
| layer4 feature_map | `[B,512,8,10]` | — |
| projected_map | `[B,256,8,10]` | — |
| position_map，纯 2D sine | `[B,256,8,10]` | — |
| visual_tokens | `[B,80,256]` | `[B,240,256]` |
| visual_pos，2D sine + camera embedding | `[B,80,256]` | `[B,240,256]` |

ResNet 使用 `ResNet18_Weights.IMAGENET1K_V1` 和 `torchvision.ops.misc.FrozenBatchNorm2d`。通过 IntermediateLayerGetter 移除 avgpool/fc，保留 layer4。BN 的 affine 与统计量均是预训练 checkpoint 的固定 buffers，调用 train() 不更新。

投影为 `Conv2d(512,256,kernel_size=1)`，不改变空间尺寸。位置编码默认 num_pos_feats=128、temperature=10000，每轴从 1 累计并归一化至 2π，y/x 各占一半通道。Hf/Wf 由卷积输出决定：224×224 → 7×7；240×320 → 8×10；255×341 → 8×11。

多相机共享 backbone 与投影。实现将相机临时放进 batch 维以并行计算，各图像独立通过卷积；不拼接图像的 RGB 通道或空间轴。每路先按 `(row,col)` 展平，再按相机序列拼接，最终顺序为 `(camera,row,col)`：top 全部 80 个 → left 全部 80 个 → right 全部 80 个。

`visual_tokens` 是纯视觉内容；`visual_pos = spatial_sine + camera_embedding[camera_id]`。同一相机的 Camera Embedding 广播到全部空间位置，不把内容与位置提前相加。Camera Embedding 可学习，因此 visual_pos 不要求限制在 [-1,1]。

## 预处理与推理复用

```python
from awbc_data import ImagePreprocessConfig, ImagePreprocessor, build_datasets

preprocess_config = ImagePreprocessConfig(
    size=(240, 320),
    mode="resize",  # 或 letterbox：等比例缩放，居中 padding
    padding_rgb=(0, 0, 0),
)
train, val = build_datasets(dataset_root, image_preprocess=preprocess_config)

# 推理时输入原始 uint8 HWC RGB 或 PIL 图像，使用同一配置：
preprocess = ImagePreprocessor(preprocess_config)
image_tensor = preprocess(rgb_image)  # [3,240,320]
```

预处理只在 Dataset/推理输入端执行一次，编码器不再归一化。mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]，RGB 顺序不变；bilinear resize，无 Center Crop。默认 resize 保持前一阶段行为；letterbox 保持宽高比并完整保留画面。

验收保存 `input_config.json`，包含预处理、编码器配置和相机顺序。后续推理可读取其中的 `image_preprocess`，构建 `ImagePreprocessConfig(**saved["image_preprocess"])`，确保一致。Dataset 旧的 `image_size` 参数仍可用；显式传入 `image_preprocess` 时以该配置为准。

## 冻结与微调

backbone_mode 可取 frozen（默认）、layer4、all。1×1 投影和 Camera Embedding 始终可训练，FrozenBatchNorm 始终固定。

```python
import torch

encoder.backbone.set_trainable("layer4")
optimizer = torch.optim.AdamW(
    encoder.parameter_groups(lr=1e-4, backbone_lr=1e-5),
    weight_decay=1e-4,
)
```

改变冻结范围后重新建立 optimizer。参数组覆盖投影、Camera Embedding 和可训练的 backbone 参数，无遗漏或重复。这里没有训练循环。

## 验收

```bash
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python \
  -m awbc_models.inspect_vision --root block_insertion_real_combined_v30

# 可选配置示例：
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python \
  -m awbc_models.inspect_vision --root block_insertion_real_combined_v30 \
  --resize-mode letterbox --height 256 --width 320 --output artifacts/awbc_vision_letterbox

lerobot-awbc/.venv/bin/python -m pytest \
  tests/test_awbc_vision.py tests/test_image_preprocessing.py tests/test_awbc_data.py -q
```

`artifacts/awbc_vision/report.json` 保存所有张量形状、类型、范围和有限值检查结果。真实预训练验收逐项核对 backbone 参数和 BN buffers；验证冻结 backbone 无梯度，投影及 Camera Embedding 有梯度，并对随机单张 RGB 输入执行 Smoke Test。

`features.png` 按相机展示真实 RGB 和 layer4 的空间特征强度（通道 RMS，每张独立缩放）。它用于检查图像内容和空间网格，不是注意力图。

默认三相机模型总参数 11,299,008（不含 BN buffers）；冻结时可训练参数 132,096，来自投影与 Camera Embedding。
