# 阶段 6：完整 Flow Matching Policy 与 Euler 采样

## 文件

新增 `policy.py`、`euler.py`、`normalization.py`、`configs/flow_matching_policy.json`、`inspect_policy.py`、`tests/test_flow_matching_policy.py` 和本说明。修改模块导出 `__init__.py`、文档入口 `README.md`、完整 Policy 的依赖声明 `requirements.txt`。

验收输出 `artifacts/awbc_policy/report.json`、`config.json`、`normalization.json`。未修改阶段 2–5 的核心模块或 Dataset/图像预处理实现，验收前后 SHA256 一致。

## 模块与统一配置

```text
FlowMatchingPolicy
 ├─ ACTVisualEncoder：共享 ResNet-18、1×1 投影、空间位置与 Camera Embedding
 ├─ ObservationEncoder：State Token + 视觉 tokens，2 层 Self-Attention
 ├─ ConditionalDiT：4 层动作 Self-Attention/Cross-Attention/AdaLN-Zero/FFN
 ├─ FlowMatchingObjective：训练样本准备、逐样本 masked MSE、权重和归一化
 ├─ action_normalizer：训练集 min/max buffers
 └─ state_normalizer：训练集 min/max buffers（use_state=True 时）
```

Euler 是独立函数，只复用编码后的 observation。完整配置复用已有网络 dataclass，由 JSON 统一加载。三个长度/步数参数独立：

| 参数 | 默认 | 含义 |
|---|---:|---|
| dit.action_horizon | 16 | 预测动作步数 |
| num_inference_steps | 8 | Euler 积分次数 |
| n_action_steps | 8 | 对外返回的执行动作步数 |

n_action_steps 必须在 [1,action_horizon]，积分次数必须为正。改变网络形状后应重新构造 Policy。from_json(..., dataset_info=...) 从元数据填充实际 action_dim、state_dim、相机数量和顺序；当前数据为 8 维动作、8 维 state、top/left/right 三相机。没有 state 字段的元数据会关闭状态分支。

## 使用

```python
from awbc_models import FlowMatchingPolicy, FlowMatchingPolicyConfig

config = FlowMatchingPolicyConfig.from_json(
    "awbc_models/configs/flow_matching_policy.json",
    dataset_info=train.source.info,
)
policy = FlowMatchingPolicy(config, stats=train.stats)

# Dataset 已预处理的训练 batch
output = policy(batch, sample_weight=None)
loss = output["loss"]
per_sample_loss = output["per_sample_loss"]
loss.backward()

# 观测编码；此入口保留梯度图
memory = policy.encode_observation(batch["observation"])

# 单次完整动作块；内部切换 eval 并关闭梯度
chunk = policy.predict_action_chunk(batch["observation"])
# chunk["action_chunk_normalized"]: [B,16,8]
# chunk["action_chunk"]:            [B,16,8]，原始动作尺度
# chunk["timings"]: 每个模块的耗时

# receding horizon：前 8 步，不包含控制循环
execution = policy.predict_action(batch["observation"])
# execution["action"]: [B,8,8]
```

predict_action_chunk/predict_action 支持 initial_noise、num_inference_steps 和 generator；不指定积分步数时使用配置。固定 observation、noise 和步数可复现结果。未提供 noise 时采样模型设备、dtype 上的标准高斯张量。指定 generator 时应匹配采样设备。

## Training Forward

```text
batch.observation.images [B,3,3,240,320]（已做 ImageNet 归一化）
    → Visual Tokens / Position [B,240,256]
batch.observation.state [B,8]（已做训练 min/max 归一化）
    → State Token [B,1,256]
    → Observation Memory / Position [B,241,256]
batch.action [B,10,8]（已归一化）
    → 输入适配 [B,16,8]，新增步数全部 mask
    → prepare_training_pair：高斯噪声、连续时间、线性路径、action-noise
    → Conditional DiT 速度 [B,16,8]
    → compute_loss：按有效动作元素平均，权重和归一化
```

forward 直接复用阶段 5 的组合 Objective，其内部仍按 prepare_training_pair → DiT → compute_loss 顺序执行一次。没有重复 loss、额外动作归一化、observation detach、Advantage 计算或 AWR。

Policy 内仅适配已有动作块长度：短于 action_horizon 时重复最后一行，并对所有新增位置设 True mask；保留已有 action_is_pad。长于 horizon 明确报错，不隐式截断。当前有效动作数仍由 Dataset 决定，例如 10 和真实尾块的 3；不会把新增的 6 或 13 个位置当作真实监督。

返回保留阶段 5 的 loss、per_sample_loss、predicted_velocity、target_velocity，以及路径和 mask 计数诊断张量。

## Euler 数学与数据流

```text
Observation → Vision（1 次）→ Observation Encoder（1 次）→ 缓存 Memory/Position/Mask
高斯噪声 x0 [B,H,A]
  → 第 k 次 DiT(xk, k/N, 同一份 Memory) → vk [B,H,A]
  → x_(k+1) = xk + vk/N
重复 N 次
  → 归一化动作 xN
  → 反归一化 1 次
  → 完整动作块
```

N 个左端点为 0/N…(N−1)/N，不在 t=1 额外调用 DiT。始终加上速度、从噪声向数据方向积分；每次积分中的 x/velocity 均保持 [B,H,A]。没有中途反归一化、裁剪、RK4 或 observation/action 序列混拼。

`sample_euler` 可以单独调用，用外部 velocity 模型验证数学。num_inference_steps 和 initial_noise shape 会检查；最终归一化动作及反归一化动作都检查 NaN/Inf。

整个预测在 torch.inference_mode 内运行，推理入口直接调用 eval()，不保存和恢复各子模块状态。若之后继续训练，调用方显式执行 policy.train()。训练 forward 保持正常 autograd。观测缓存只在本次动作块内复用，下一次 predict 会重新编码新观测。

## Dataset / 原始观测与归一化

Dataset 的真实 observation 接口为 images、state。encode_observation 和 predict 方法均接收该预处理后的字典，不再次归一化。

原始单次观测使用单独入口：

```python
raw_observation = {
    "images": {camera_name: uint8_rgb_image for camera_name in config.camera_names},
    "state": raw_state_vector,
}
observation = policy.prepare_observation(raw_observation)
result = policy.predict_action_chunk(observation)
```

prepare_observation 复用阶段 1/2 的 ImagePreprocessor：RGB、配置化 bilinear resize 或 letterbox、除以 255、ImageNet mean/std，不做 Center Crop。相机按配置顺序排列，不按 dict 插入顺序。该入口处理单个环境，返回 [1,N_cam,3,H,W] 和可选 [1,state_dim]。

State 使用已有训练 min/max；关闭状态分支时既不要求 state，也不创建虚假 state。

Dataset 现有 normalize/denormalize 是 CPU Dataset 实例方法，因此新建薄封装 MinMaxNormalizer，公式逐项相同，并将 min/max 注册为 buffers，便于随设备移动及进入 state_dict：

```text
normalize(x) = 2*(x-min)/(max-min)-1
constant dimension -> 0
denormalize(z) = (z+1)*(max-min)/2+min
```

常量判定阈值沿用 Dataset 的 span>1e-8。没有重新统计 min/max，不做关节差分、控制单位转换或夹爪重编码。动作反归一化输出 float32，所有维度沿用原始顺序。混合精度建议由外部 autocast 控制，使统计 buffers 继续保持 float32；CPU bfloat16 autocast 路径已测试。

构造 Policy 必须提供 train.stats，或配置 normalization_stats_path；缺失时明确报错。验收将 Dataset 的同一份 train.stats 保存到 normalization.json，然后通过配置路径读入 Policy。它包含训练 episode IDs、状态全量真实帧及动作真实 transition 的统计信息；验证沿用训练统计。动作反归一化与 Dataset 逐值完全一致。

如果提供 observation["visual_padding_mask"]，会直接交给 Observation Encoder。现有 letterbox 仍没有自动生成视觉 token mask，保持之前阶段的行为；动作 mask 和视觉 mask 不混用。

## State Dict 与统计量恢复

```python
# 常规权重与 buffers 保存，不提供 checkpoint 管理系统
config.normalization_stats_path = "/absolute/path/to/normalization.json"
config.to_json("config.json")
torch.save(policy.state_dict(), "policy.pt")

restored = FlowMatchingPolicy(FlowMatchingPolicyConfig.from_json("config.json"))
restored.load_state_dict(torch.load("policy.pt", weights_only=True))
```

配置中的相对统计路径按配置文件目录解析。配置保存预处理与相机顺序；min/max 同时保存在 state_dict 的 buffers 中，加载会恢复原始统计值。测试故意改变构造时统计文件，确认 load_state_dict 后仍恢复原始 min/max 与推理结果。没有实现 EMA、实验管理或 checkpoint 生命周期管理。

## 参数量

默认实际 8 维状态/动作、三相机配置：

| 模块 | 总参数 | 可训练参数 |
|---|---:|---:|
| Vision Encoder | 11,299,008 | 132,096 |
| Observation Encoder | 1,582,080 | 1,582,080 |
| Conditional DiT | 6,814,984 | 6,814,984 |
| Objective / Normalizers / Euler | 0 | 0 |
| 总计 | 19,696,072 | 8,529,160 |

Normalizer 和 FrozenBatchNorm 的 buffers 不计入参数量。冻结 ResNet 后，视觉部分仅投影与 Camera Embedding 可训练。

## 测试与真实验收

```bash
# 本阶段独立测试
lerobot-awbc/.venv/bin/python -m pytest tests/test_flow_matching_policy.py -q

# 联合回归
lerobot-awbc/.venv/bin/python -m pytest \
  tests/test_flow_matching_policy.py tests/test_flow_matching.py \
  tests/test_awbc_dit.py tests/test_awbc_observation.py \
  tests/test_awbc_vision.py tests/test_image_preprocessing.py -q

# 真实预训练视觉模型 + Dataset batch
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python \
  -m awbc_models.inspect_policy --root block_insertion_real_combined_v30
```

结果：19 项本阶段测试通过，75 项联合回归通过；真实 batch 完整前向、backward、高斯采样、固定噪声采样、state_dict 保存/加载均通过。真实数据输入 [2,10,8]，有效步数 [10,3]，默认训练适配后 [2,16,8]；一次 training forward loss 为 2.018554，所有梯度有限。

原始 AdaLN-Zero/零输出头初始化下，真实第一次 backward 的 DiT 有非零梯度，上游梯度已连接但数值为零。测试专用的非零 Head/Cross gate 用例确认完整 Policy 内 Observation Encoder、State Position、视觉投影和 Camera Embedding 的实际非零梯度；没有增加 Optimizer 或训练循环。冻结 ResNet 始终无梯度。

Euler Dummy 恒定速度测试在 N=1/4/8/16 下都满足 xN=x0+v，并逐次检查实际输入时间、积分方向和调用次数，避免未训练真实 DiT 的零速度让数学测试虚假通过。额外覆盖 fp32/fp16/bfloat16 的 Euler 张量、随机 Generator、CPU bfloat16 autocast、非法 noise/步数、NaN/Inf、仅编码一次、推理 eval 模式、确定性、单次反归一化、无裁剪、执行切片、无状态分支、原始 RGB 预处理和统计配置恢复。

## 推理耗时

以下为本机 CPU、PyTorch 2.11.0、4 个 CPU threads、B=2、三相机 240×320、float32，完成一次热身后的单次测量。是接口验收数据，不是平均值或部署延迟保证。计时从预处理后的 observation 开始，不含 MP4 解码或 prepare_observation。

| Euler 次数 | Vision ms | Observation ms | Euler 总计 ms | 完整生成 ms |
|---|---:|---:|---:|---:|
| 1 | 53.63 | 2.88 | 2.86 | 59.74 |
| 4 | 53.81 | 2.88 | 10.32 | 67.39 |
| 8 | 53.84 | 3.38 | 22.28 | 79.88 |
| 16 | 53.21 | 2.96 | 40.61 | 97.18 |

8 次积分中的每次 DiT 前向约 2.47–3.08 ms，反归一化约 0.011 ms。report.json 保存所有逐次耗时和调用计数。每次生成始终为 Vision=1、Observation=1、DiT=N。GPU/MPS 计时会在边界同步设备，保证计时覆盖实际执行。

## 规格说明

网络结构、阶段 5 数学及 sample_weight reduction 均未修改。为了接入实际 10 步 Dataset，在 Policy 的训练输入处补齐到配置 horizon 并全部 mask；没有改变监督数量。正常推理仍预测完整 16 步，执行接口只返回前 8 步。

新增 prepare_observation 作为显式原始输入入口，避免混淆已经归一化的 Dataset batch 与原始相机/state。推理直接进入 eval，模式切换由调用方显式管理。输出增加 timings，仅用于观测性能，不改变网络计算。

简化后去掉了重复的积分次数校验、网络间 hidden/camera/统计维度预检查、Euler 每步速度形状检查以及归一化 buffers 的逐次自动迁移。配置和数据结构由现有管线保证，buffer 随 policy.to(device) 移动；保留明确要求的噪声形状、执行长度、缺失统计量、有限值检查及动作块长度边界。

当前权重尚未经过完整任务训练，零速度头下归一化采样结果保留初始噪声；本阶段验证接口、数学、梯度和数值，没有验证任务成功率。没有硬件控制、AWR/IQL、Scheduler、ODE 高阶方法或完整训练器。

阶段 6 到此结束，等待检查。
