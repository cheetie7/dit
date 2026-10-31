# 阶段 7：Flow Matching Baseline 训练框架

阶段 8 的 10 步 AWR-FM、预计算 RoboMeter 标签接口与来源审查见 [AWR_FM.md](AWR_FM.md)。阶段 7 的下列实测记录保留当时的 16 步配置；新默认配置已按阶段 8 改为 10 步，旧 checkpoint 按其保存的 Policy 配置恢复。

## 交付与范围

新增独立 `awbc_training` 包，复用 `awbc_data` 和 `awbc_models`，没有重写模型。当前训练是普通、无 sample weight 的 Flow Matching baseline，不读取 advantage 作为权重，不接入 AWR/IQL/STEAM/RoboMeter。

按用户最后要求，只保存普通 checkpoint，没有 best checkpoint、best loss 选择或相关状态。

唯一旧代码接口扩展是 `FlowMatchingPolicy.forward(batch, sample_weight=None, *, noise=None, timestep=None)`：增加两个可选参数传给现有 Objective，使验证可重复。已有调用完全兼容，数学和网络结构不变。Euler、ResNet、Observation Encoder、DiT、Objective 与 Dataset 均未改动。

## 数据泄漏审查

当前原始数据有 146 条完整轨迹、51,996 帧、单个任务。原 Dataset 已按轨迹拆分，数值列按 frame_index 排序，图像通过 episode 的共享视频时间偏移对齐；action 从观测时刻 t 开始切块，终止观测不作有效 transition。动作块不跨轨迹。

训练入口在完整轨迹上用 seed=42 做 90/10 划分，默认按 episode 元数据的 tasks 分层。当前只有一组任务，不虚构实验条件分组。其他元数据分组键可配置；单例组只能保留在训练侧，不能把同一轨迹复制进验证。

实际结果：

| 数据 | 轨迹数 | 10 步 chunk 数 |
|---|---:|---:|
| Train | 131 | 4,758 |
| Validation | 15 | 494 |
| 合计 | 146 | 5,252 |

与先前顺序留出末尾轨迹的 4731/521 个窗口不同，是因为现在按 seed=42 随机划分了完整轨迹。新划分的 train/val episode ID 无交集，窗口不跨原轨迹；完整 ID 与取样索引写入每次实验的 split.json。

训练统计只使用这 131 条训练轨迹：state 含真实终止观测，action 排除终止行与 padding。当前 action 统计包含 46,983 个真实 transition，state 包含 47,114 帧。验证和 Policy 的 Action/State Normalizer 复用同一份 min/max；推理使用保存的同一份统计。

旧的 `artifacts/awbc_data/normalization.json` 和 `artifacts/awbc_policy/normalization.json` 来自此前顺序划分，仍保留原样；新实验统计写入新目录，不覆盖旧统计。

当前训练和验证均不使用随机图像增强，沿用共享 RGB resize/ImageNet 预处理。每次 batch 解码由原 reader 检查视频时间误差；入口另保存 batch_report.json 和对齐检查。真实 Dataset 的 action_is_pad 原样经过 Policy 的 horizon 适配传给 Objective，补到 16 的位置全部 mask，按真实动作数量平均。

## 模块和文件

- `config.py`：复用已有 JSON/dataclass 风格，支持分节配置与 CLI dotted overrides。
- `data.py`：轨迹分层划分、训练统计、审查 manifest 与诊断子集。
- `optim.py`：AdamW 参数分组与 warmup/cosine。
- `runtime.py`：设备、AMP、batch 移动、CPU/CUDA/MPS RNG 保存恢复。
- `ema.py`：完整 Policy EMA。
- `checkpoint.py`：临时文件 + 原子替换、latest/定期 checkpoint 与保留策略。
- `logging.py`：train.log、metrics.csv、metrics.jsonl 和可选 WandB。
- `trainer.py`：训练、验证、累积、更新、恢复逻辑。
- `train.py`：入口、独立实验目录、配置/统计/元信息保存。
- `configs/baseline.json`：参考完整训练配置；本次未启动。
- `configs/smoke.json`：3 次更新的短验收。
- `configs/overfit.json`：固定真实 batch 的 24 次诊断。
- `tests/test_training.py`：训练系统独立测试。

同时更新 `awbc_models/policy.py` 的可选参数，以及 `awbc_models/README.md` 的文档入口。

## 配置与命令

从仓库根目录运行。参考完整配置使用 epochs=200、batch=16、workers=4、AdamW lr=1e-4、weight_decay=1e-4、warmup=500、min_lr_ratio=0.01、clip=1.0、EMA decay=0.999、BF16 autocast。这只是初始配置，不代表最优参数，也尚未做 GPU batch size 调优。

```bash
# 短 smoke：CPU、真实数据、BF16、3 次更新
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python -m awbc_training.train \
  --config awbc_training/configs/smoke.json

# 从普通 latest checkpoint 继续；max_updates 是累计更新上限
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python -m awbc_training.train \
  --resume outputs/<run>/checkpoints/latest_checkpoint.pt \
  --set training.max_updates=4

# 单个固定 batch 的短诊断
TORCH_HOME=artifacts/torch lerobot-awbc/.venv/bin/python -m awbc_training.train \
  --config awbc_training/configs/overfit.json

# 完整配置的使用示例；本次没有执行该命令
python -m awbc_training.train --config awbc_training/configs/baseline.json \
  --set dataset.root=/path/to/dataset \
  --set training.device=cuda \
  --set training.batch_size=16 \
  --set optimizer.learning_rate=0.0001 \
  --set scheduler.warmup_steps=500
```

Batch Size、LR、Warmup 分别是独立字段。网络参数通过 policy 小节覆盖已有 Policy 配置，例如 `--set policy.vision.backbone_mode=layer4`，或同时调整三个编码器的隐藏维度。实际 action_dim/state_dim/相机字段由 Dataset metadata 读取。

路径按仓库根目录使用。每次新实验创建 `outputs/<name>-<Shanghai timestamp>/`；显式指定已存在的 --run-dir 会自然报错，不覆盖其他实验。resume 只追加当前实验日志，另存 resume_config，不覆盖初始配置、统计与划分记录。

## Training Loop

```text
按 epoch seed 构建轨迹窗口的固定随机顺序
 → 从 checkpoint 的 batch cursor 开始
 → policy.train()
 → autocast 下 Policy(batch)，默认无 sample weight
 → loss 按当前累积窗口真实样本数缩放
 → scaled backward
 → 窗口结束：unscale → clip → AdamW step → scaler update
 → 只有实际更新成功：global_step++ → scheduler step → EMA update
 → zero_grad
 → 原始/EMA 固定条件验证
 → 正常 checkpoint 保存
```

总计划更新数为 `ceil(ceil(train_samples/batch_size)/accumulation)*epochs`。完整默认数据对应 298 次更新/epoch、59,600 次计划更新。

等长 minibatch 的累积缩放就是 loss/G；最后不足 G 个 batch 或最后 batch 较小时，按当前窗口实际样本数分配比例，等价于对窗口内全部样本求均值。这样不会把短窗口的梯度额外缩小。

clip 只处理 requires_grad=True 的参数，FP16 先 unscale。返回的裁剪前 gradient norm 被记录。非有限 loss 或非 AMP 的非有限 norm 直接报错；FP16 溢出由 GradScaler 跳过更新并记录，不推进 scheduler/EMA/global_step。测试模拟跳过一次更新验证该逻辑。

max_updates 是诊断停止条件，不改变计划 LR 曲线。中断只发生在完成一个累积窗口之后，checkpoint 不需要保存未完成的梯度。

## Optimizer 与 Scheduler

只遍历一次 named_parameters，跳过冻结参数，不重复加入参数。Backbone 解冻时使用独立 backbone_learning_rate，默认 1e-5；其他组默认 1e-4。Bias 和一维 normalization 参数默认无 weight decay，可配置关闭该分组规则。

当前冻结 ResNet 的分组：

| 分组 | 参数量 | Smoke 初始 LR |
|---|---:|---:|
| Visual Projection + Camera Embedding，decay | 131,840 | 1e-4 |
| Visual bias，no decay | 256 | 1e-4 |
| Observation Encoder，decay | 1,575,168 | 1e-4 |
| Observation bias/norm，no decay | 6,912 | 1e-4 |
| DiT，decay | 6,791,168 | 1e-4 |
| DiT bias，no decay | 23,816 | 1e-4 |

总参数 19,696,072，可训练 8,529,160，冻结 11,166,912。启动时打印总数、各组参数量、实际初始 LR；数据也保存在 optimizer.json。

LambdaLR 使用更新计数。warmup 初始系数为 1/warmup_steps，随后升到 1；之后 cosine 下降至 min_lr_ratio。没有 warmup 时从 base LR 开始。日志记录当前这次更新实际使用的 LR，scheduler 状态进入 checkpoint。

## AMP、EMA 和 Validation

参数始终保持 FP32，计算用 autocast。支持 FP32、BF16 和 FP16；FP16 使用 GradScaler，BF16 不使用。设备不可用、CUDA BF16 不支持或不支持的 MPS FP16 scaler 路径会给出明确提示。Loss 聚合继续由阶段 5 保持 FP32。

EMA 从完整 Policy 复制初始化，始终是有效权重；每次成功更新后按 `ema = decay*ema + (1-decay)*policy` 更新可训练参数，冻结参数和全部 buffers 直接复制。保存 EMA 权重、decay、更新次数。没有在 minibatch 上提前更新。

验证使用 eval + inference_mode，顺序固定、batch size 固定、Validation Seed 固定。使用专用 CPU Generator 生成可复现 noise/time，再搬到设备；不消耗训练全局 RNG。原始 Policy 和 EMA 对每个 batch 复用完全相同的 noise/time，分别记录 val/loss、val/ema_loss。没有参数或 EMA 更新，也没有 advantage weighting。

## Checkpoint 和恢复

普通 checkpoint 包含 Policy、EMA、AdamW、scheduler、GradScaler、epoch、下一 batch cursor、global_step、统计量、划分/子集索引、完整实验及 Policy 配置、Python/NumPy/Torch CPU/CUDA/MPS RNG，以及部分 epoch 的 loss 计数。固定 batch 诊断还保存其固定 noise/time。

每个 epoch 结束或短诊断停止时保存 latest_checkpoint.pt；按配置周期额外保存 epoch_XXXX.pt，并保留最近 keep_last_n 个普通定期文件。可用 checkpoint.update_interval 额外按更新间隔保存。保存使用同目录临时文件写完再 os.replace 原子替换。

resume 恢复全部状态和下一 minibatch 位置，使用 epoch seed 直接重建剩余顺序，不重新解码已消耗 batch。独立 DataLoader Generator 避免 iterator 创建消耗训练 RNG。当前没有随机图像增强，因此 worker 数变化不改变样本内容；未来加入随机增强时需另行设计 worker/样本级 RNG。

为保证当前 data cursor 和 scheduler 的含义，精确 resume 保持训练 seed、batch_size、accumulation 和原计划更新数一致；max_updates、日志设置和 workers 可独立修改。划分/统计量变化明确报错，不静默换数据。

推理可从普通 checkpoint 加载原始或 EMA Policy：

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
policy = FlowMatchingPolicy(
    FlowMatchingPolicyConfig.from_json(run_dir / "policy_config.json"),
    stats=checkpoint["normalization_stats"],
)
policy.load_state_dict(checkpoint["ema"]["policy"] if checkpoint["ema"] else checkpoint["policy"])
policy.eval()
```

没有实现 best checkpoint。

## 日志与实验目录

```text
outputs/<unique_run>/
  config.json                  # 初始实际实验配置
  policy_config.json           # 解析 metadata 后的完整模型配置
  normalization_stats.json     # 此次训练轨迹统计
  split.json                   # 完整轨迹 ID、chunk 数、诊断索引
  metadata.json                # Git hash / PyTorch / source SHA
  optimizer.json               # 参数量与分组
  batch_report.json            # 全部 batch 张量 shape/dtype/range
  train.log
  metrics.csv
  metrics.jsonl
  result.json
  checkpoints/latest_checkpoint.pt
  checkpoints/epoch_0001.pt
```

记录 train loss、原始/EMA val loss、各组 LR、裁剪前 norm、global step、batch/data 时间、epoch 时间与 CUDA 峰值显存。WandB 可关闭，默认配置为 offline，安装或连接失败不影响本地训练；本次真实验收关闭 WandB，另用测试验证失败后的本地 fallback。

当前根 Git 仓库尚无 HEAD commit，因此 metadata 中 git_commit=null，同时保存模型源文件 SHA256，而不是虚构提交号。

## 测试与实测结果

```bash
lerobot-awbc/.venv/bin/python -m pytest tests/test_training.py -q

lerobot-awbc/.venv/bin/python -m pytest \
  tests/test_training.py tests/test_flow_matching_policy.py tests/test_flow_matching.py \
  tests/test_awbc_dit.py tests/test_awbc_observation.py tests/test_awbc_vision.py \
  tests/test_image_preprocessing.py -q
```

14 项训练系统测试、89 项联合回归通过。覆盖：轨迹/统计/视频对齐、参数分组、CPU 三种精度、FP16 scaler 状态恢复、梯度累积与末尾短窗口、warmup/cosine、EMA 数学及 buffers、验证复现且不影响训练 RNG、原子保存/保留、中途恢复与连续训练参数逐值一致、固定输入恢复一致、跳过更新不推进 scheduler/EMA、WandB fallback、CLI overrides。

实际运行：

- `outputs/stage7_smoke-20261030-211600-248975`：真实模型、完整三相机、BF16 CPU，先 3 次成功更新并验证，然后加载 latest 恢复到 global_step=4。Scheduler 与 EMA 更新数均连续，EMA updates=4。恢复后 val/loss=1.338781、val/ema_loss=1.356548。
- `outputs/stage7_worker_smoke-20261030-212305-017972`：num_workers=1，真实数据完成一次更新、原始/EMA 验证及保存，验证训练 DataLoader 的进程路径。
- `outputs/stage7_overfit_one_batch-20261030-211845-098430`：同一真实 batch、固定 noise/time、关闭 dropout，24 次 FP32 诊断更新。每步训练 loss 从 1.183674 降到第 24 步的 0.064150；只是可优化性检查，不代表泛化或任务成功率。

短 smoke 每 epoch 仅使用 12 个训练 chunk、4 个验证 chunk；完整轨迹划分和统计仍来自完整训练集，未混入验证集。Overfit 使用 2 个固定 chunk，不进行长时间训练。没有启动 200 epoch 的参考完整训练。

当前设备仅 CPU，未发现 CUDA 或可用 MPS。CPU FP32/BF16/FP16 前向反向已测，真实 BF16 smoke 有限值正常；本机 BF16 比 FP32 诊断更慢。实际 GPU 峰值显存不可测，不将本地日志中的占位 0 当作 CUDA 显存结果。GPU 上 batch=16 的适配情况、CUDA AMP 和 MPS 路径尚需目标硬件验证。

汇总结果在 `artifacts/awbc_training/report.json`。
