# 阶段 3：机器人状态与 Observation Transformer 验收

## 文件

新增 `awbc_models/observation.py`（状态投影与融合模块）、`awbc_models/inspect_observation.py`（真实数据验收）、`tests/test_awbc_observation.py`（独立测试）、本说明。

修改 `awbc_models/__init__.py` 导出新模块，`awbc_models/README.md` 增加阶段 3 入口。验收生成 `artifacts/awbc_observation/report.json` 与 `input_config.json`。

阶段 2 的 `awbc_models/vision.py` 和 Dataset 核心 `awbc_data/dataset.py` 未改动。验收前后 SHA256 一致：

```text
vision.py   620f0c09eb9b78ffeac70a00a49059b7806893b84f67baf191baa4e283ca8ecc
dataset.py  7cdf65bc8202eb040c6c3caa3f0c3dcef4d552f1ad414f7780fa8629ee578849
```

## 数据接口和实际形状

状态沿用阶段 1 的 `batch["observation"]["state"]`。原始字段为 `observation.state`，metadata shape 为 `[8]`，对应数据审查报告中的既有 8 维状态接口；未追加独立保存的 velocity/effort，未改变状态语义。直接使用 Dataset 的训练集 min/max 归一化结果，不做第二次归一化。

配置的 state_dim 在验收入口从 `train.source.info["features"]["observation.state"]["shape"][0]` 读取，不使用 ACT 中的硬编码 14。

B=2、三相机、图像 240×320 的真实结果：

| 阶段 | 张量形状 |
|---|---|
| state | `[2,8]` |
| Linear(8,256) + unsqueeze | `[2,1,256]` |
| visual_tokens / visual_pos | `[2,240,256]` |
| observation_tokens / observation_pos | `[2,241,256]` |
| 两层 Encoder 后 memory | `[2,241,256]` |
| use_state=False 后 memory | `[2,240,256]` |

State Token 位于序列第一位，后续视觉顺序保持阶段 2 的 `(camera,row,col)`。拼接沿 dim=1，不拼接特征通道，不池化空间位置。

## 使用

```python
from awbc_models import (
    ACTVisualEncoder, VisualEncoderConfig,
    ObservationEncoder, ObservationEncoderConfig,
)

vision = ACTVisualEncoder(VisualEncoderConfig(
    num_cameras=len(train.source.cameras),
))
encoder = ObservationEncoder(ObservationEncoderConfig(
    state_dim=train.source.info["features"]["observation.state"]["shape"][0],
))
features = vision(batch["observation"]["images"])
output = encoder(**features, state=batch["observation"]["state"])

memory = output["observation_memory"]
positions = output["observation_pos"]
padding_mask = output["observation_padding_mask"]
```

诊断入口 `encoder.prepare_tokens(...)` 返回进入第一层 attention 之前的 observation_tokens、observation_pos、observation_padding_mask。关闭状态分支使用 `ObservationEncoderConfig(use_state=False)`，不用配置 state_dim，也不创建任何虚假的状态或状态参数。

## 位置与 Attention

State Position 是独立的可学习 `Parameter(torch.randn(1,1,256))`，沿 batch 扩展。视觉位置完整保留阶段 2 的二维 sine 与 Camera Embedding，既不覆盖也不重新编号。

```text
observation_tokens = cat([Linear(state).unsqueeze(1), visual_tokens], dim=1)
observation_pos    = cat([state_pos.expand(B,-1,-1), visual_pos], dim=1)
```

两层都使用当前层的内容构造 attention，而非仅在第一层加位置：

```text
Q = current_tokens + observation_pos
K = current_tokens + observation_pos
V = current_tokens
```

`nn.MultiheadAttention` 内部仍会执行自身的 Q/K/V 线性投影。这里描述的是传入 MHA 的张量。State Token 与所有有效视觉 tokens 做全序列 self-attention；没有 causal mask。

每层具体结构：

```text
MultiheadAttention(hidden=256, heads=8, dropout=0.1, batch_first=True)
  → residual + dropout → LayerNorm
Linear(256,1024) → GELU → dropout(0.1) → Linear(1024,256)
  → residual + dropout → LayerNorm
```

共两层，参数独立，采用 post-norm；Encoder 之外不追加末尾 LayerNorm。Transformer 矩阵参数用 Xavier uniform 初始化。默认 training 模式有 dropout；推理调用 eval() 关闭 dropout。

## 输出与 mask

统一输出：

```python
{
    "observation_memory": ...,       # [B,N+1,256]，无状态时 [B,N,256]
    "observation_pos": ...,          # 与 memory 同形状
    "observation_padding_mask": ..., # None 或 [B,N+1]/[B,N] bool
}
```

传入 visual_padding_mask 时，True 表示屏蔽，False 表示有效。启用 State Token 后为其前置一个 False，视觉 mask 顺序不变。未传入视觉 mask 时返回 None。

mask 用作每一层 MHA 的 key_padding_mask，屏蔽位置不作为 Key/Value 被有效 token 聚合。被屏蔽位置的输出本身不保证为零；下一阶段应继续使用返回的同一个 mask。use_state=False 时每个样本应至少有一个有效视觉 token。

目前 Dataset 默认 resize 不产生图像 padding mask，所以真实默认输出为 None。验收另外构造一个视觉 token mask 检查传播，不能把 action_is_pad 当作视觉 mask。阶段 2 的 letterbox 也尚未自动生成 token mask，此阶段不修改该视觉模块。

下一阶段可直接用 memory+positions 构造 Cross-Attention 的 Key，用 memory 构造 Value，并传入返回的 padding mask。本阶段只保留接口，没有实现下一阶段模块。

## 测试与真实数据验收

```bash
# 阶段 3 独立测试
lerobot-awbc/.venv/bin/python -m pytest tests/test_awbc_observation.py -q

# 视觉、预处理与 Observation Encoder 联合回归
lerobot-awbc/.venv/bin/python -m pytest \
  tests/test_awbc_observation.py tests/test_awbc_vision.py tests/test_image_preprocessing.py -q

# 使用已缓存的官方预训练权重，跑真实 Dataset batch
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python \
  -m awbc_models.inspect_observation --root block_insertion_real_combined_v30
```

结果：阶段 3 独立测试 10 passed；联合回归 26 passed；真实数据验收成功。B=2 的 memory 数值范围约 [-4.3063,3.9437]，所有报告张量均无 NaN/Inf。

覆盖内容：State Token 编码、49+1 token 拼接、独立状态位置参数、每层 Q/K/V 的真实输入、两层结构与参数、mean().backward() 连通性、随机加权损失的非零梯度、关闭状态分支、mask 屏蔽效果与梯度、state 对视觉 memory 的影响、visual_pos 确实参与 attention。

注意，post-norm 的最后一个 LayerNorm 初始 weight=1/bias=0，会让 memory.mean() 接近常量。测试执行了要求中的 mean().backward() 并检查梯度存在且有限；另用 `(memory*random_readout).mean()` 验证各可训练参数、输入视觉内容和位置的非零梯度。真实 batch 验收也使用后一种诊断目标，仅做一次 backward，没有 optimizer step。

真实集成验收确认 frozen ResNet 无梯度，1×1 投影、Camera Embedding、State Projection、State Position 和两层 Transformer 均有有限非零梯度。逐参数梯度范数及数据来源 episode/frame 保存在 report.json。

## 与 ACT 原版的差异及原因

参考 [ACT detr_vae.py](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py) 和 [transformer.py](https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py)：

- 保留 ACT 的状态 Linear 投影、独立状态位置编码、逐层 Q/K 加位置、Value 保持内容及 post-norm 残差结构。
- 按当前数据读取 state_dim=8，替换 ACT 中硬编码的 14；不扩充 Dataset 输入。
- 按本阶段配置使用 2 层、8 头、FFN=1024、GELU；官方模块的激活默认是 ReLU，层数等通过其配置传入。
- 使用 batch-first `[B,N,D]`，匹配现有视觉接口；官方核心 attention 使用 sequence-first。数学上的 token 交互方式相同。
- 只前置一个 State Token。按阶段范围省略 latent/CVAE、Action Query、Decoder 和动作输出，直接返回 Encoder memory。
- 接收阶段 2 的逐相机序列及 Camera Embedding，不改回 ACT 的沿图像宽度拼接方式，保持已经确认的视觉接口。

阶段 3 到此结束，未接入 DiT、Flow Matching 或完整 Policy 训练。
