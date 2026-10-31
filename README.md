# DiT Flow Policy

用于离线机器人示教的独立 PyTorch 项目：ResNet-18 → Observation Transformer → Conditional DiT，支持普通 Flow Matching（FM）和 Advantage-Weighted Flow Matching（AWR-FM）。默认预测 **10 步动作块**，执行长度单独配置为 8；不依赖 LeRobot Python 包、机器人 SDK 或 RoboMeter 在线模型。

## 安装

推荐 Python 3.12，支持 Python ≥3.10。先按目标设备安装匹配的 PyTorch / torchvision，再安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
# 可选的 WandB 日志
python -m pip install -e '.[wandb]'
```

本机验证版本：Python 3.12、PyTorch 2.11.0、torchvision 0.26.0。CUDA、MPS 或 CPU 由 training.device 配置；GPU 上的 AMP 支持以运行时检查为准。

## 项目结构

```text
dit-flow-policy/
├── pyproject.toml              # 安装、依赖、命令行入口
├── train.py                    # python train.py 与 awbc-train 等价
├── configs/                    # FM、AWR-FM、smoke、overfit 配置
├── src/
│   ├── awbc_data/               # 轨迹读取、预处理、窗口与标签对齐
│   ├── awbc_models/             # Vision、Observation、DiT、Objective、Euler、Policy
│   └── awbc_training/           # Trainer、EMA、优化器、日志、恢复与推理加载
├── tests/                      # 合成数据测试，不需要真实数据或下载预训练权重
├── tools/                      # 已有 RoboMeter 标注流程与数据审计脚本
└── docs/                       # 数据、架构、训练说明与历史验收记录
```

真实数据、模型权重和历史实验不打包进项目。可将数据放在 `data/`，也可直接传外部绝对路径。模型配置随安装包发布，运行不依赖旧工作区；旧模块名保留，便于恢复已有 checkpoint。

## 先验证安装

```bash
pytest -q
awbc-train --help
# 仅使用人工观测、动作和 Advantage，FM/AWR-FM 各更新 3 次
awbc-smoke
```

`awbc-smoke` 不读取真实机器人数据、不下载 ImageNet 权重，输出保存在当前目录的 `outputs/` 与 `artifacts/`。

## 数据检查与短测试

以下命令只执行 3 次 optimizer update、验证并保存 checkpoint。首次使用预训练 ResNet-18 时，torchvision 会下载 ImageNet 权重；离线服务器需提前缓存相应权重。

```bash
awbc-inspect-data --root /path/to/block_insertion_real_combined_v30_advantage \
  --require-advantage --shuffle-episodes --seed 42

awbc-train --config configs/smoke.json \
  --set dataset.root=/path/to/block_insertion_real_combined_v30_advantage
```

`smoke.json` 使用 CPU/BF16、小 batch、0 个 worker；可分别覆盖 device 与 AMP。各配置中的相对数据、输出及自定义配置路径均相对于**命令运行目录**，CLI 覆盖遵守同一约定。跨目录或服务器运行时推荐使用绝对路径。

## FM 与 AWR-FM

下面是完整训练的启动接口，项目整理时没有执行完整训练：

```bash
# 普通 FM baseline
awbc-train --config configs/fm.json \
  --set dataset.root=/path/to/dataset \
  --set training.device=cuda \
  --set training.batch_size=16

# AWR-FM：已经算好的 10 步窗口标签，来源信息需与实际标注相符
awbc-train --config configs/awr_fm.json \
  --set dataset.root=/path/to/scored_dataset \
  --set advantage_weighting.metadata_path=/path/to/advantage_metadata.json
```

标注已在 Parquet `annotation.*` 中时不必另传标签表。若使用外部 JSONL/CSV/Parquet 表，增加 `--set advantage_weighting.labels_path=/path/to/chunk_advantages.jsonl`。

来源文件格式见 `configs/advantage_metadata.template.json`。真实标注检查记录见 [数据审计](docs/data_audit/README.md)：当前下载数据已确认完整，精确 checkpoint 版本与评分模型训练数据范围仍未知。模板不会伪造这些值，训练入口仍要求明确版本和来源审查记录。

训练默认关闭 WandB，仅写本地日志；需要时设置 `logging.use_wandb=true` 和 `logging.wandb_mode=offline/online`。AdamW、warmup/cosine、AMP、梯度累积/裁剪、EMA、验证、checkpoint 间隔都可通过 JSON 或 `--set key=value` 修改。

可以直接按总优化器更新次数训练：

```bash
awbc-train --config configs/fm.json \
  --set dataset.root=/path/to/dataset \
  --set training.num_steps=50000
```

`num_steps` 设置后优先于 `num_epochs`，同时控制训练终点和 warmup/cosine 的总长度。未设置时继续按 epoch 预算计算；`max_updates` 仍仅用于提前停止诊断，不改变学习率调度长度。Epoch 是遍历一次训练集；step 是一次成功的 optimizer update，梯度累积的多个 minibatch 合起来才算一步，也不等于机器人动作步或 Flow 时间 t。

默认 tqdm 显示全局更新进度、epoch、loss、LR、速度和 ETA；AMP 跳过更新时进度不增加，resume 从保存的 global_step 接续。验证有单独 batch 进度。设置 `logging.progress_bar=false` 可关闭，CSV/JSONL/WandB 日志不受影响。

## 断点恢复

```bash
awbc-train --resume outputs/<run>/checkpoints/latest_checkpoint.pt
# 继续一个设置了短停止点的诊断实验
awbc-train --resume outputs/<run>/checkpoints/latest_checkpoint.pt \
  --set training.max_updates=4
```

恢复 optimizer、scheduler、scaler、EMA、随机状态和数据位置。相同实验保持原数据划分、batch/累积与总调度；10 步新实验不覆盖以前的 16 步实验。旧 checkpoint 按保存的模型配置恢复，checkpoint 内已含视觉骨干权重，恢复和推理不需要再次下载 ImageNet 权重。

每次新实验使用唯一目录，保存完整配置、模型配置、划分、训练集归一化统计、来源记录、CSV/JSONL 日志，以及 latest 和定期 checkpoint。按现有要求不实现 best checkpoint。

## 离线推理接口

```python
import numpy as np
from awbc_training.inference import load_policy

policy = load_policy("outputs/<run>/checkpoints/latest_checkpoint.pt",
                     use_ema=True, device="cpu")
# 三个数组均为 uint8 RGB，键名及顺序以 policy.config.camera_names 为准
raw_observation = {
    "images": {
        "observation.images.top_camera": top_rgb,
        "observation.images.left_camera": left_rgb,
        "observation.images.right_camera": right_rgb,
    },
    "state": np.asarray(robot_state, dtype=np.float32),
}
observation = policy.prepare_observation(raw_observation)
result = policy.predict_action_chunk(observation)
actions = result["action_chunk"]  # [1, 10, action_dim]，已反归一化
```

如果实验未启用 EMA，使用 `use_ema=False`。这只提供离线张量推理，不包含机器人执行逻辑。

更多说明：[数据与标签](docs/DATA.md)、[模型与损失](docs/ARCHITECTURE.md)、[训练与实验](docs/TRAINING.md)、[整理与验证记录](docs/PROJECT.md)。
