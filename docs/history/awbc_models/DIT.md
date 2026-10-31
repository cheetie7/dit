# 阶段 4：Conditional DiT 网络验收

## 文件变更

新增：

- `awbc_models/dit.py`：配置、连续时间嵌入、调制函数、DiT Block、Final Layer 和主网络。
- `awbc_models/configs/conditional_dit.json`：全部网络超参数，action_dim 留空供数据元信息填入。
- `awbc_models/inspect_dit.py`：真实 Observation Encoder 输出的集成验收。
- `tests/test_awbc_dit.py`：独立前向、初始化、梯度、mask 和条件依赖测试。
- 本文件。

修改 `awbc_models/__init__.py` 导出 DiT 接口、`awbc_models/README.md` 增加阶段 4 文档入口。验收生成 `artifacts/awbc_dit/report.json` 和实际配置 `config.json`。

阶段 2/3 及 Dataset 核心文件未改动，验收前后 SHA256 均一致：

```text
vision.py       620f0c09eb9b78ffeac70a00a49059b7806893b84f67baf191baa4e283ca8ecc
observation.py  94647785ad5150b0461eb09c93af021fba521ede8954b07e3baa94299c6e5080
dataset.py      7cdf65bc8202eb040c6c3caa3f0c3dcef4d552f1ad414f7780fa8629ee578849
```

## 接口与配置

```python
from awbc_models import ConditionalDiT, ConditionalDiTConfig

config = ConditionalDiTConfig.from_json(
    "awbc_models/configs/conditional_dit.json",
    action_dim=train.source.info["features"]["action"]["shape"][0],
)
dit = ConditionalDiT(config)
velocity = dit(
    noisy_action=noisy_action,          # [B,H,A]，由调用方提供
    timestep=timestep,                  # [B]，连续浮点数
    observation_memory=memory,          # [B,N,D]
    observation_pos=positions,          # [B,N,D] 或 None
    observation_padding_mask=mask,      # bool[B,N] 或 None
)
```

默认配置：H=16、D=256、4 个独立 Block、8 头（每头 32 维）、FFN=1024、dropout=0.1、GELU；时间 Fourier 维度 128、时间 MLP 中间维度 256。JSON 同时支持配置 max_period=10000、LayerNorm eps=1e-6。沿用已有 dataclass 配置风格，from_json 只是文件加载入口。

action_dim 从数据元信息或配置读取，当前实际为 8；代码不固定为单元测试示例中的 7。传入的 H 必须与配置 action_horizon 一致，D 必须与上游 memory/position 的隐藏维度一致。修改这些值时重新构造网络即可。

## 完整结构与形状

实际集成 B=2、A=8、H=16、N=241：

| 模块 | 输入 | 输出 |
|---|---|---|
| Action Linear | `[2,16,8]` | `[2,16,256]` |
| 可学习 Action Position，相加 | `[1,16,256]` | `[2,16,256]` |
| 连续时间 Fourier | `[2]` | `[2,128]` |
| 时间 MLP：Linear → SiLU → Linear | `[2,128]` | `[2,256]` |
| 每个 Block 的调制 Linear | `[2,256]` | `[2,2304]`，拆为 9×`[2,256]` |
| Self-Attention，8 头 | `[2,16,256]` | `[2,16,256]` |
| Cross-Attention，8 头 | Q 长 16，K/V 长 241 | `[2,16,256]` |
| FFN：256 → 1024 → 256 | `[2,16,256]` | `[2,16,256]` |
| 以上 Block ×4 | `[2,16,256]` | `[2,16,256]` |
| Final LN + shift/scale | `[2,16,256]` | `[2,16,256]` |
| Velocity Linear：256 → 8 | `[2,16,256]` | `[2,16,8]` |

动作序列保持逐步 token，没有展平成一个动作向量。Action Position 是 `[1,H,D]` 的可学习零初始化 Parameter，独立于视觉的二维位置。

每个 Block 顺序为：Pre-LayerNorm → 时间调制 → Self-Attention → 门控残差；随后 Pre-LayerNorm → 时间调制 → Cross-Attention → 门控残差；最后 Pre-LayerNorm → 时间调制 → 两层 GELU FFN → 门控残差。LayerNorm 不使用可学习 affine，采用 DiT 的 eps=1e-6。Attention 内 dropout=0.1，FFN 中间与输出各有 dropout=0.1。

## Q / K / V

Self-Attention 三者均为调制后的当前动作内容：

```text
h = modulate(LN(x), shift_sa, scale_sa)
Q = K = V = h
```

Cross-Attention 使用：

```text
Q = modulate(LN(x), shift_ca, scale_ca)
K = observation_memory + observation_pos
V = observation_memory
```

如果未提供 observation_pos，则 K=memory。两种 attention 都使用 batch_first=True。Observation Mask 为 True 时屏蔽对应 K/V，None 表示无屏蔽。至少应保留一个有效的 observation token。不把 observation 与 action 序列拼接，不把 memory 汇聚为全局条件，也不向时间调制分支输入 observation。

## AdaLN-Zero 和 Final Layer

调制函数：`x*(1+scale[:,None,:])+shift[:,None,:]`。

每个 Block 使用 SiLU → Linear(D,9D)，依次得到 self-attention、cross-attention 和 FFN 各自的 shift、scale、gate。每个残差是 `x + gate[:,None,:]*branch_output`，没有额外 sigmoid。

所有 Block 的调制 Linear 权重和 bias 均为零，因此初始化时三个 gate 都为零，整个 Block 严格保持输入不变。Final Layer 使用无 affine 的 LN 和 SiLU → Linear(D,2D)，其调制同样零初始化。最后独立的 Velocity Head Linear(D,A) 权重和 bias 为零，因此初始网络输出严格为零。

其他 Linear 使用 Xavier uniform/零 bias；时间 MLP 权重使用 std=0.02 的正态参数初始化。这里的参数初始化不涉及动作噪声采样。

Velocity 输出契约是速度场张量，没有 tanh、clip、动作解码或 epsilon/方差分支。当前未训练，初始零输出不能用于执行机器人动作。

## 连续时间嵌入

对 k=0…63，频率为 `exp(-log(10000)*k/64)`，把浮点 timestep 与频率相乘，再拼接 cos/sin，得到 128 维向量。t=0、t=1 及中间小数都可计算，时间不转成整数，不额外乘离散扩散步数。

再经 `Linear(128,256) → SiLU → Linear(256,256)` 得到所有 Block 共用的时间条件。该条件只依赖 timestep；各个 Block 使用各自的调制参数。

## 参数量

以下为实际 A=8、H=16 配置的参数量，所有参数均可训练：

| 模块 | 参数量 |
|---|---:|
| Action Projection | 2,304 |
| Action Position | 4,096 |
| Timestep Embedder | 98,816 |
| DiT Blocks ×4 | 6,576,128 |
| Final Layer 调制 | 131,584 |
| Velocity Head | 2,056 |
| **总计** | **6,814,984** |

每个 Block 为 1,644,032 个参数：Self-Attention 263,168；Cross-Attention 263,168；FFN 525,568；AdaLN 调制 592,128；三个非 affine LayerNorm 无参数。完整计数保存在验收 JSON，`dit.parameter_counts()` 可直接查询。

## 测试结果

```bash
# DiT 独立测试
lerobot-awbc/.venv/bin/python -m pytest tests/test_awbc_dit.py -q

# 与已有视觉、Observation Encoder、预处理一起回归
lerobot-awbc/.venv/bin/python -m pytest \
  tests/test_awbc_dit.py tests/test_awbc_observation.py \
  tests/test_awbc_vision.py tests/test_image_preprocessing.py -q

# 真实 observation 集成
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python \
  -m awbc_models.inspect_dit --root block_insertion_real_combined_v30
```

结果：DiT 独立测试 8 passed；联合回归 34 passed；真实 batch 集成成功，所有输出无 NaN/Inf。

独立测试包括：B=4/H=16/A=7/N=50 基础形状、A=8/H=10、全部超参数从 JSON 读取、连续时间边界和小数、attention 的实际 Q/K/V 输入、初始 Block 恒等与输出严格零、Observation Padding Mask、诊断更新后条件依赖与有效梯度。

零初始化梯度测试只在 pytest fixture 中执行 4 次小规模确定性张量回归更新。第 1 次 backward 中 Head 有非零梯度，Block 的 attention/调制可能为零；后续调制 gate 逐步打开，测试确认每个 Block 的 Self-Attention、Cross-Attention、FFN、AdaLN，以及时间 MLP、动作投影与动作位置都有有效梯度。所有梯度均检查有限值。

在更新后的 eval 网络上固定 action/timestep，改变 memory、position 或 timestep 均改变输出。更改被 mask 的 memory/position 不改变输出，而且被 mask 输入的梯度为零。该检查在门控打开后完成，避免初始零输出使测试虚假通过。

真实验收使用阶段 2 预训练 ResNet、阶段 3 Observation Encoder，得到 `[2,241,256]` memory。按阶段 4 默认 H=16 得到 `[2,16,8]` velocity；再按 Dataset 的真实 H=10 配置另一个 DiT，得到 `[2,10,8]`。对默认网络只执行一次诊断 backward，未执行 optimizer step。完整形状、数值范围、模块参数量和来源 episode/frame 均在 report.json。

## 规格说明与参考差异

网络结构和初始化遵循本阶段规格，没有增加其他策略组件。Final Layer 的调制与最终 Linear 分成两个具名模块，数学运算顺序与规格相同。

当前 Dataset 的 chunk 仍为 10，阶段 4 明确要求默认 H=16，因此网络默认按 16 实现，并允许配置 H=10。真实 16 步验收使用确定性的外部给定输入张量，仅测试接口；没有补写 Dataset、没有增加监督动作，也没有生成带噪训练样本。

为遵守阶段范围，测试输入使用确定性的 sin/linspace 张量，未采样高斯动作噪声。多次 optimizer 更新仅限单元测试中的诊断，不提供训练循环或 Flow Matching Loss。

参考 [原始 DiT](https://github.com/facebookresearch/DiT/blob/main/models.py) 的时间 Fourier/MLP、AdaLN-Zero 和零输出初始化；按本阶段为三个残差分支扩展为 9D 调制。参考 [ACT Transformer](https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py) 的内容/位置分离 Cross-Attention。检查了 [LeRobot DiTFlow](https://github.com/danielsanjosepro/lerobot_policy_ditflow/blob/main/src/lerobot_policy_ditflow/modeling_ditflow.py)，但不引入其全局 observation 条件、采样器或 Policy 包装。

阶段 4 到此结束，等待检查。
