# 阶段 5：Conditional Flow Matching Objective

实现位于 `flow_matching_objective.py`，是无可训练参数的 Objective。直接调用已有 ConditionalDiT，不更改阶段 2/3/4 或 Dataset，不包含 Optimizer、Scheduler、训练循环或 ODE Solver。

## 分离的接口

```python
from awbc_models import FlowMatchingObjective

objective = FlowMatchingObjective()

# 1. 噪声、时间、插值、目标速度
pair = objective.prepare_training_pair(action, noise=None, timestep=None)

# 2. 速度预测：沿用阶段 4 的真实参数名称
prediction = dit(
    noisy_action=pair["noisy_action"],
    timestep=pair["timestep"],
    observation_memory=observations["observation_memory"],
    observation_pos=observations["observation_pos"],
    observation_padding_mask=observations["observation_padding_mask"],
)

# 3. 独立 loss reduction
loss_output = objective.compute_loss(
    predicted_velocity=prediction,
    target_velocity=pair["target_velocity"],
    action_padding_mask=action_is_pad,  # bool[B,H]，可选
    sample_weight=sample_weight,        # float[B]，可选
)
loss = loss_output["loss"]
per_sample_loss = loss_output["per_sample_loss"]
```

也保留组合调用的便利入口，不创建 Policy 或修改 DiT：

```python
result = objective(
    dit=dit,
    action=action,
    **observations,
    action_padding_mask=action_is_pad,
    sample_weight=sample_weight,
    noise=None,
    timestep=None,
)
```

组合入口不对 observation memory 或 position 执行 detach/no_grad，梯度图原样保留。独立的 `compute_loss` 不依赖 DiT，也不采样，便于后续独立组织训练前向。

Objective 不假设 H 或 A；它们来自输入 action。模型的 action_horizon/action_dim 必须与 action 对齐。当前 Dataset 的动作块仍为 10 步；可以配置 H=10 的 DiT 直接调用，也可以在调用端补到 H=16 并把新增位置标为 padding。Objective 不自动补齐或截断动作。

## 采样与数学定义

```text
noise = randn_like(action)
timestep = rand(B, device=action.device)
t = timestep.to(action.dtype).view(B,1,1)
noisy_action = (1-t)*noise + t*action
target_velocity = action-noise
```

时间方向为 t=0 是噪声，t=1 是数据。目标速度不依赖 t。每个样本独立采样 float32 连续时间，范围 [0,1)，外部可精确传入 0 和 1。

noise 保持 action 的设备、形状和 dtype；外部 noise 会移到 action 的设备和 dtype。插值系数转换为 action.dtype，保证模型输入维持 float32、float16 或 bfloat16。模型与 observation 的 dtype 配置或 autocast 由调用方负责。

传入外部 noise/timestep 时不重新采样对应变量。`linear_flow_path` 是该准备步骤的独立函数版本。

Objective 原样使用 Dataset 已归一化的 action，不重新归一化、不反归一化、不改变动作控制语义。训练和验证继续复用训练集统计量。

## 逐样本 loss 与 padding

先把 prediction 和 target 转成 float32，再做残差与平方。半精度目标值为 300 时，MSE=90000，超过 float16 上限，但仍在这里稳定计算为 float32。

无 padding 时，每样本等价于 `squared_error.mean(dim=(1,2))`。有 padding 时，True 表示无效动作，分母为真实有效动作步数乘动作维度 A：

```text
error = predicted_velocity.float() - target_velocity.float()
error[padding] = 0
per_sample_loss = error.square().sum((1,2)) / (valid_steps*A)
```

在平方前清零被 mask 的残差，与有限值情况下的平方后乘 mask 数学等价，还能防止极大的 padded prediction 在平方时溢出。padding 的直接误差项梯度为零。

任何样本的有效步数为零时抛出明确 ValueError，即使该样本权重为零也不允许静默纳入平均。`action_padding_mask` 只用于误差监督；与交给 Cross-Attention 的 `observation_padding_mask` 分开。

## Sample Weight

没有权重时：`loss = per_sample_loss.mean()`。

提供 [B] 权重时：

```text
loss = sum(sample_weight * per_sample_loss) / sum(sample_weight)
```

权重必须一维、有限、非负且总和大于零；错误形状、负数、NaN、Inf 和全零权重均报错。所有权重为同一个正数时，直接使用普通均值分支，保证与无权重结果精确一致。因为已确认权重和为正，不需要用 epsilon 掩盖全零输入。

per_sample_loss 始终保留未加权版本。权重应用于样本，不在动作元素层面定义独立权重。Objective 不计算 Advantage，不实现 exp(A/beta) 或 AWR/STEAM。

此处采用阶段 5 完整规格明确要求的“权重和归一化”，不是前期 AWBC 文档中的 weighted batch mean。

## 返回值

`prepare_training_pair` 返回四个张量：

| 字段 | 形状 | dtype |
|---|---|---|
| noise | `[B,H,A]` | action.dtype |
| timestep | `[B]` | 默认 float32 |
| noisy_action | `[B,H,A]` | action.dtype |
| target_velocity | `[B,H,A]` | action.dtype |

`compute_loss` 返回：

| 字段 | 形状 | 语义 |
|---|---|---|
| loss | `[]` | float32，最终 loss |
| per_sample_loss | `[B]` | float32，未加权逐样本 loss |
| weighted_per_sample_loss | `[B]` | weight*per_sample_loss，尚未除权重和 |
| sample_weight | `[B]` | float32；未提供时为 1 |
| valid_action_count | `[B]` | int64，真实动作步数 |
| elementwise_loss | `[B,H,A]` | float32，padding 为零 |

组合 forward 返回以上全部张量，并增加 predicted_velocity `[B,H,A]`。所有 loss 张量保留 autograd 图，需要长期记录时由调用方 detach。

## 测试与真实验收

```bash
lerobot-awbc/.venv/bin/python -m pytest tests/test_flow_matching.py -q

lerobot-awbc/.venv/bin/python -m pytest \
  tests/test_flow_matching.py tests/test_awbc_dit.py \
  tests/test_awbc_observation.py tests/test_awbc_vision.py tests/test_image_preprocessing.py -q

TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python \
  -m awbc_models.inspect_flow_matching --root block_insertion_real_combined_v30
```

结果：22 项 Objective 独立测试通过；56 项联合回归通过；真实 Dataset → 预训练 ResNet → Observation Encoder → DiT → Objective 前向及 backward 成功。

测试覆盖规格中的噪声 shape `[4,16,7]`、标准正态分布和复现、batch 内独立时间、t=0/1/0.5 及中间路径、速度符号与时间无关性、零误差与误差增长、逐样本均值、权重归一化和常量权重等价、所有非法权重、padding 分母、全 padding 报错、零权重梯度、极大 padded 值、fp32/fp16/bfloat16、真实 DiT 调用、禁止隐式广播、观测梯度连通性。

另在单元测试中使用现有 torch optimizer 做 4 次小规模诊断更新，验证原始 AdaLN-Zero/零 Head 初始化逐步打开后，完整 Vision→Observation→DiT→Objective 路径具有实际非零梯度。State Projection、State Position、Observation Attention、视觉投影、Camera Embedding 和 DiT Cross-Attention 均获得非零梯度；冻结 ResNet 无梯度，所有梯度有限。没有增加优化器实现或训练器，这段更新只存在于测试函数中。

真实验收包含完整 chunk 和真实尾块，Dataset 输入 `[2,10,8]`，有效步数 `[10,3]`。适配至默认 H=16 后，分别有 6 和 13 个 padding 步，所有新增位置排除监督。原 Dataset 内容和采样逻辑未改动。

固定 seed=42 的本次真实结果：per_sample_loss 约 `[1.102080,0.838232]`；普通 batch loss 为 `0.970156`；示例权重 `[1,2]` 经权重和归一化后为 `0.926181`。这些权重仅用于验收接口，不代表从原始数据获得了 AWR 标签。

初始 DiT 的 Head 权重为零，所以真实首次 backward 的 Head 梯度非零，memory/position 梯度连通但数值为零。测试通过上述诊断更新及非零初始化补充用例确认这不是 detach 导致的。真实验收没有 optimizer step。

`artifacts/awbc_flow_matching/report.json` 包含返回张量形状、范围、实际 loss、mask、梯度及受保护文件 SHA256。所有报告张量与已有梯度均有限。

## 文件变更与规格差异

新增 `awbc_models/flow_matching_objective.py`、`awbc_models/inspect_flow_matching.py`、`tests/test_flow_matching.py`、本说明；更新 `awbc_models/__init__.py` 和 `README.md`。

`vision.py`、`observation.py`、`dit.py`、`awbc_data/dataset.py` 的 SHA256 与阶段 5 开始前一致。

核心数学与聚合遵循完整规格。实现细节差异只有：mask 在平方前处理以避免无效位置溢出；MSE 使用 float32；在明确要求的分离接口之外提供组合 forward 及更多诊断字段。这些不改变目标函数。

阶段 5 到此结束，等待检查。
