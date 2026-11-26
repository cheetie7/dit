# DiT Flow Policy

使用共享 ResNet-18、Observation Transformer 和 Conditional DiT 学习机器人动作块，支持普通 Flow Matching、RoboMeter Advantage 加权及 IQL critic 导出的 Q−V 加权。

推荐流程是先独立训练 Q/V critic，导出训练窗口的 Advantage，再训练 DiT policy。已有 critic checkpoint 时，从 Advantage 导出开始即可。

## 1. 安装训练环境

项目支持 Python ≥3.10，纯训练建议 Python 3.12。需要加载 simulation-simple 时使用 Python 3.11。GPU 训练应先安装与本机驱动、CUDA 环境匹配的 PyTorch 和 torchvision，再安装项目。

~~~bash
git clone https://github.com/cheetie7/dit.git
cd dit

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print("MPS:", torch.backends.mps.is_available())'
awbc-train --help
python -m awbc_training.iql --help
python -m awbc_training.iql_dense --help
awbc-eval-sim --help
pytest -q
~~~

CPU 可运行数据检查和合成测试，pytest 无需 MuJoCo。首次使用预训练 ResNet-18 会下载 ImageNet 权重，可显式指定缓存目录。WandB 是可选依赖。

~~~bash
export TORCH_HOME="$PWD/artifacts/torch"
python -m pip install -e '.[wandb]'
~~~

后续命令默认从 DiT 根目录执行，且已经激活训练环境。配置中的相对路径相对于命令运行目录，实际数据建议使用绝对路径。新实验的输出目录必须尚不存在。

## 2. 配置与入口

| 配置 | 用途 | 入口 |
|---|---|---|
| configs/fm.json | 普通 FM policy | awbc-train |
| configs/awr_fm.json | RoboMeter Advantage 加权 policy | awbc-train |
| configs/iql_critic.json | 稀疏成功奖励 Q/V critic | python -m awbc_training.iql train |
| configs/iql_dense_critic.json | 进度塑形奖励 Q/V critic | python -m awbc_training.iql_dense train |
| configs/iql_fm.json | 用 sparse/dense critic 的 Q−V 加权 policy | awbc-train |
| configs/iql_dense_awbc.json | 融合 Dense critic 与 RoboMeter 两路权重 | awbc-train |
| configs/eval_sim_block_insertion.json | 仿真映射 | awbc-eval-sim |
| configs/smoke.json、overfit.json | 小规模诊断 | awbc-train |

直接编辑 JSON，或通过多次 --set key=value 覆盖。数字、列表、布尔值和 null 按 JSON 解析。Critic 使用顶层 num_steps，policy 使用 training.num_steps，均表示成功的 optimizer update 次数。梯度累积的多个 microbatch 合计一次 policy 更新。Policy 指定 num_steps 后优先于 num_epochs。

## 3. 准备数据与标签

输入为 LeRobot v3.0 Parquet 数值列和时间戳对齐的 RGB MP4。数据、终止标签、RoboMeter 评分和模型权重由调用方准备。

~~~text
dataset/
├── meta/
│   ├── info.json
│   └── episodes/**/*.parquet
├── data/**/*.parquet
├── videos/**/*.mp4
├── episode_end_labels.jsonl
└── boundary_progress_iql.jsonl
~~~

训练入口按完整轨迹划分 train/val，状态和动作的 min-max 统计只从训练轨迹计算。图像按配置 resize 或 letterbox，使用 ImageNet mean/std。验证与推理复用训练统计。

### 状态、动作和窗口

当前 iql_critic、iql_dense_critic、iql_dense_awbc 配置选择本项目 block insertion 数据中的左臂七个关节角及夹爪状态，动作选择左臂七个关节命令及夹爪命令。

~~~json
{
  "dataset": {
    "state_indices": [0, 1, 2, 3, 4, 5, 6, 28],
    "action_indices": [0, 1, 2, 3, 4, 5, 6, 7],
    "train_stride": 10
  }
}
~~~

索引依据该数据集 meta/info.json 的原始 44 维状态和 16 维动作顺序。其他机器人需按各自字段定义配置。列表顺序会保留，省略或设为 null 时使用完整列。模型维度自动匹配切片结果。

iql_fm.json 默认使用完整状态和动作，消费左臂 critic 标签时也要填写上述索引。

Critic 的 action_horizon 和 policy 的 policy.dit.action_horizon 定义窗口名义长度 H。dataset.train_stride 定义训练起点间隔，null 时等于 H。验证 stride 等于 H。起点 t、有效长度 L 的样本使用动作 [t,t+L)，下一观测是同一轨迹 t+L 帧。短尾块补齐到 H，padding 不参与 loss。

### 终止标签

IQL 需要 episode_end_labels.jsonl，每条轨迹一行。

~~~json
{"episode_index": 12, "success": true, "end_type": "terminated", "end_frame_index": 347}
~~~

end_type 为 terminated 或 truncated。episode_index 使用保存后的 ID，采集 seed 可能与其不同。数据约定 N 帧包含 N−1 个有效 transition；end_frame_index 必须是最后有效动作执行后的第 N−1 帧。成功终止的最后一个有效动作给 +1，折算到窗口起点为 gamma 的 L−1 次方。真正终止时 bootstrap_mask=0，有有效下一观测的截断为 1。

本地曾对 300 条仿真轨迹补入复制末帧，补丁版为 60,509 帧和 60,209 个有效动作。该补丁数据是独立准备的产物，需连同终止标签一起使用。使用其他原始导出时，先核实末动作后的观测语义。

### 原始进度标签

Dense IQL 和 Dense AWBC 读取 boundary_progress_iql.jsonl。

~~~json
{"episode_index": 12, "relative_frame_index": 0, "progress_raw": 0.2}
{"episode_index": 12, "relative_frame_index": 20, "progress_raw": 0.6}
{"episode_index": 12, "relative_frame_index": 347, "progress_raw": 0.78}
~~~

按 episode_index 和 relative_frame_index 对应数据的 episode_index、frame_index。使用 progress_raw，保留实际末帧分数。旧评分中的 progress_used 可能把终点强制设为 1，Dense 路径读取原始值。

每条轨迹必须覆盖首末帧，相邻评分之间按帧号线性插值，允许下降。原始标签间隔 20 帧与训练 horizon、stride 独立。复制出来的末帧应显式增加同一观测的原始分数。已有补丁数据的 boundary_progress_iql.jsonl 已补齐；缺失边界、重复键、非有限值和越界帧都会报错。

Dense critic 的数据奖励和 TD target 为：

~~~text
reward_dense = gamma * progress(t+L) - progress(t)
reward = reward_sparse + shaping_scale * reward_dense
discount = gamma ** L
TD target = reward + discount * bootstrap_mask * V(next_observation)
~~~

Dense 项使用固定 gamma，bootstrap 折扣使用实际有效长度 L。原始 RoboMeter 评分工具和额外依赖见 [标注工具说明](tools/labeling/README.md)。训练使用预先计算好的标签文件。

## 4. 独立训练 critic

设置实际数据路径和新输出目录。下面 sparse 和 dense 两种训练选择一种执行。

~~~bash
data=/absolute/path/to/block_insertion_rgb_640x480_300_iql
mkdir -p outputs

test -f "$data/meta/info.json"
test -f "$data/episode_end_labels.jsonl"
~~~

基础 IQL 使用稀疏成功奖励。

~~~bash
critic_run="$PWD/outputs/critic_left_run1"

python -m awbc_training.iql train \
  --config configs/iql_critic.json \
  --set dataset.root="$data" \
  --set labels_path="$data/episode_end_labels.jsonl" \
  --set device=cuda \
  --run-dir "$critic_run"
~~~

Dense IQL 增加原始进度塑形奖励。

~~~bash
critic_run="$PWD/outputs/dense_critic_left_run1"

python -m awbc_training.iql_dense train \
  --config configs/iql_dense_critic.json \
  --set dataset.root="$data" \
  --set labels_path="$data/episode_end_labels.jsonl" \
  --set progress_path="$data/boundary_progress_iql.jsonl" \
  --set device=cuda \
  --run-dir "$critic_run"
~~~

### 可选：分别训练 Q 和 V 的视觉编码器

两份 critic 配置默认 `vision_encoder_mode="shared_frozen"`，保持共享冻结 ResNet-18，兼容缺少此字段的旧 checkpoint。第一步解冻实验设置 `vision_encoder_mode="separate_trainable"`：Q1/Q2 共享一个可训练 ResNet-18，V 使用另一个独立可训练 ResNet-18，保留 FrozenBatchNorm；每个相机的特征仍池化至 2×2，再按相机顺序展平并拼接 state。两个编码器均使用 `pretrained` 指定的现有初始化方式。

设置数据目录后，分别启动 sparse 或 dense 实验：

~~~bash
export DATASET_ROOT="<DATASET_ROOT>"

python -m awbc_training.iql train \
  --config configs/iql_critic.json \
  --set dataset.root="$DATASET_ROOT" \
  --set labels_path="$DATASET_ROOT/episode_end_labels.jsonl" \
  --set vision_encoder_mode=separate_trainable \
  --run-dir outputs/iql_separate_trainable

python -m awbc_training.iql_dense train \
  --config configs/iql_dense_critic.json \
  --set dataset.root="$DATASET_ROOT" \
  --set labels_path="$DATASET_ROOT/episode_end_labels.jsonl" \
  --set progress_path="$DATASET_ROOT/boundary_progress_iql.jsonl" \
  --set vision_encoder_mode=separate_trainable \
  --run-dir outputs/iql_dense_separate_trainable
~~~

V loss 仅更新 V 编码器和 V head；Q loss 仅更新共享 Q 编码器和两个 Q head。target Q 编码器与 heads 从在线 Q 复制初始化，参数统一按 `target ← (1−ρ) target + ρ online` 更新，目标计算无梯度。奖励、TD target 和 MLP 不变。模式保存在 checkpoint 配置及 Advantage metadata 中；恢复时必须保持模式一致。导出命令与下节相同，计算 `min(target_Q1,target_Q2)−V` 时分别使用 target Q 编码器和 V 编码器。

按当前配置的 3 相机、8 维 state、8 维 action、horizon=10、隐藏层 [256,256]，参数量如下；状态、相机数或 horizon 改变时，head 参数量随之变化。

| 分支 | separate_trainable 可训练参数 | shared_frozen 可训练参数 |
| --- | ---: | ---: |
| V 编码器 + V head | 12,808,129 | 1,641,217 |
| 共享 Q 编码器 + Q1/Q2 heads | 14,495,426 | 3,328,514 |
| target Q 编码器 + heads | 0 | 0 |

每个 ResNet-18 编码器含 11,166,912 个参数；新模式在线可训练参数合计 27,303,555，target Q 保存 14,495,426 个冻结参数。FrozenBatchNorm 的固定统计量不计入可训练参数。

### 第二步：Spatial Softmax 视觉特征

顶层 `vision_feature_mode` 默认为 `avg_pool_2x2`，保持第一步的特征处理和旧 checkpoint 兼容。第二步使用 `separate_trainable` 与 `spatial_softmax` 的组合，开启新 run：

~~~bash
export DATASET_ROOT="<DATASET_ROOT>"

python -m awbc_training.iql train \
  --config configs/iql_critic.json \
  --set dataset.root="$DATASET_ROOT" \
  --set labels_path="$DATASET_ROOT/episode_end_labels.jsonl" \
  --set vision_encoder_mode=separate_trainable \
  --set vision_feature_mode=spatial_softmax \
  --run-dir outputs/iql_spatial_softmax

python -m awbc_training.iql_dense train \
  --config configs/iql_dense_critic.json \
  --set dataset.root="$DATASET_ROOT" \
  --set labels_path="$DATASET_ROOT/episode_end_labels.jsonl" \
  --set progress_path="$DATASET_ROOT/boundary_progress_iql.jsonl" \
  --set vision_encoder_mode=separate_trainable \
  --set vision_feature_mode=spatial_softmax \
  --run-dir outputs/iql_dense_spatial_softmax
~~~

相机按数据集顺序沿 RGB 通道拼接为 `[B,3N,H,W]`；ResNet-18 的首层 RGB 卷积核重复 N 次并除以 N，其他 stride、FrozenBatchNorm 和预训练初始化方式保持不变。layer4 输出的每个通道独立在空间位置上做温度 1 的 softmax，然后在 `[-1,1]` 网格上计算期望坐标。x 从左到右递增，y 从上到下递增；1024 维向量顺序为 `(x0,y0,x1,y1,...)`，再通过 `Linear(1024,50) → LayerNorm(50) → tanh`。

50 维视觉特征直接拼接归一化 state：8 维 state 时 V 输入为 58 维；8 维 action、horizon=40 时 Q 输入为 `58 + 40×8 + 40 = 418` 维。上述命令沿用配置中的 horizon、stride、MLP、奖励和 TD 目标设置。

Q1/Q2 共享 ResNet 和投影，V 的 ResNet 和投影独立；target Q 的 ResNet、投影及两个 heads 一起 Polyak 更新。checkpoint 和 Advantage metadata 保存 `vision_feature_mode`，恢复时禁止改变模式。旧 checkpoint 缺少该字段按 `avg_pool_2x2` 解释。Spatial Softmax checkpoint 的恢复及导出仍使用原有 sparse/dense 子命令，导出的 Advantage 仍为 `min(target_Q1,target_Q2)−V`，policy 标签格式不变。

默认 10000 次更新、batch 16。CPU 设置 device=cpu。需要 stride=2 时增加 --set dataset.train_stride=2。训练步数及保留 checkpoint 数量可直接编辑 critic JSON，例如：

~~~json
{
  "num_steps": 5000,
  "batch_size": 16,
  "checkpoint_interval": 1000,
  "checkpoint_keep_last_n": 5
}
~~~

checkpoints/ 下保存 latest_checkpoint.pt 和最近指定数量的 step_XXXXXXXX.pt，训练结束时也保存。可使用某个中间 checkpoint 导出 Advantage。首次小规模验收可设置 num_steps=3、batch_size=2、validation_interval=3；无预训练缓存时可设 pretrained=false。

## 5. 从已有 checkpoint 导出 Advantage

使用与 critic 奖励模式对应的模块。仅遍历 checkpoint 所对应的训练窗口，输出原始 min(target_q1,target_q2)−V，保留正负值。

~~~bash
checkpoint="$critic_run/checkpoints/latest_checkpoint.pt"

# 基础 IQL checkpoint
python -m awbc_training.iql export \
  --checkpoint "$checkpoint" --output "$critic_run/advantages.jsonl" --device cuda
~~~

Dense IQL checkpoint 则执行：

~~~bash
python -m awbc_training.iql_dense export \
  --checkpoint "$checkpoint" --output "$critic_run/advantages.jsonl" --device cuda
~~~

若 checkpoint 保存了旧数据路径，可覆盖路径，继续读取同一份数据和标签。

~~~bash
python -m awbc_training.iql_dense export \
  --checkpoint "$checkpoint" --output "$critic_run/advantages.jsonl" --device cuda \
  --set dataset.root="$data" \
  --set labels_path="$data/episode_end_labels.jsonl" \
  --set progress_path="$data/boundary_progress_iql.jsonl"
~~~

产物为 advantages.jsonl、advantages.metadata.json。Metadata 保存 critic 标识、数据划分、奖励来源、horizon、stride、状态和动作选择，policy 加载时会核对。

## 6. 训练 IQL-FM policy

Sparse 和 Dense critic 的标签都可接入 iql_fm.json。下面配置与前述单左臂 critic 的 H=10、stride=10 一致。

~~~bash
actor_run="$PWD/outputs/iql_fm_left_run1"

awbc-train --config configs/iql_fm.json \
  --set dataset.root="$data" \
  --set 'dataset.state_indices=[0,1,2,3,4,5,6,28]' \
  --set 'dataset.action_indices=[0,1,2,3,4,5,6,7]' \
  --set dataset.train_stride=10 \
  --set policy.dit.action_horizon=10 \
  --set advantage_weighting.labels_path="$critic_run/advantages.jsonl" \
  --set advantage_weighting.metadata_path="$critic_run/advantages.metadata.json" \
  --set training.device=cuda \
  --set training.num_steps=10000 \
  --run-dir "$actor_run"
~~~

如果 critic 使用其他 horizon、stride 或列选择，上面也需保持一致。验证使用普通 FM loss，标签只需覆盖训练窗口。

IQL-FM 的默认权重配置为 normalization=none、beta=1、min_weight=0.1、max_weight=10。每个窗口使用 exp(A/beta) 并裁剪，按一次 optimizer update 全部样本的总权重归一化，梯度累积共享总分母。

希望启动命令简短时，将参数写入独立 JSON。

~~~bash
cp configs/iql_fm.json configs/iql_fm.local.json
# 编辑数据路径、列索引、窗口、标签路径和 training.num_steps
awbc-train --config configs/iql_fm.local.json
~~~

configs/*.local.json 由 Git 忽略。修改既有配置即可使用，不需要修改训练源码。

当前 policy 默认预测 10 步、执行前 8 步。policy.n_action_steps 控制执行块长度，想一次执行全部预测动作可设为 horizon。动作先在归一化空间生成，再按 checkpoint 中保存的动作统计反归一化。

| Policy 字段 | 用途 |
|---|---|
| training.num_steps | 总更新预算，优先于 num_epochs |
| training.batch_size | 每个 microbatch 的样本数 |
| training.gradient_accumulation_steps | 每次更新累积多少个 microbatch |
| validation.update_interval | 普通 FM 验证间隔 |
| checkpoint.update_interval | checkpoint 保存间隔 |
| checkpoint.keep_last_n | 保留最近多少个定期文件 |
| logging.use_wandb / wandb_mode | 是否启用 WandB 及 online/offline 模式 |

### 其他 policy 模式

~~~bash
awbc-train --config configs/fm.json \
  --set dataset.root="$data" --set training.device=cuda
~~~

普通 FM 可以直接读取数据，状态/动作默认使用完整维度。awr_fm.json 使用已算好的 RoboMeter chunk Advantage，支持 Parquet annotation 或外部标签及其来源信息，见 [数据说明](docs/DATA.md)。

iql_dense_awbc.json 读取 Dense critic 标签，并额外融合原始无折扣进度差。设置 advantage_weighting.labels_path、metadata_path 和 dense_awbc.progress_path。默认固定 RoboMeter scale=1，关闭衰减；scale_start=0 为纯 critic 对照。该模式 critic 权重为 exp(min(beta_q*A, log(critic_weight_cap)))，其中 beta_q 乘在 A 上，IQL-FM 则使用 A/beta。

## 7. 结果与断点恢复

~~~text
<actor_run>/
├── config.json / policy_config.json
├── normalization_stats.json / split.json
├── advantage_provenance.json / advantage_stats.json
├── metrics.jsonl / metrics.csv / train.log
├── advantage_analysis/
├── checkpoints/
│   ├── latest_checkpoint.pt
│   └── step_XXXXXXXX.pt
└── sim_eval/                         # 启用仿真验证时生成
~~~

恢复 policy 会沿用原输出目录及 optimizer、scheduler、EMA、随机状态、数据位置。

~~~bash
awbc-train --resume "$actor_run/checkpoints/latest_checkpoint.pt"
~~~

恢复 Dense critic：

~~~bash
python -m awbc_training.iql_dense train \
  --resume "$critic_run/checkpoints/latest_checkpoint.pt"
~~~

基础 critic 使用 awbc_training.iql。Policy 同一次恢复需保持原总训练预算、batch 和梯度累积设置。窗口、状态/动作选择、数据划分或标签版本变化时启动新实验。恢复和推理直接使用 checkpoint 的视觉权重与统计。

## 8. 配置可选仿真环境

仿真使用私有仓库 [Songan-Lab/simulation-simple](https://github.com/Songan-Lab/simulation-simple)，需要 Python 3.11 和 uv。先安装 GitHub CLI，用具有读取权限的账号登录。

~~~bash
gh auth status --hostname github.com
# 未登录时执行
gh auth login --hostname github.com --git-protocol https --web
gh repo view Songan-Lab/simulation-simple

mkdir -p external
gh repo clone Songan-Lab/simulation-simple external/simulation-simple
git -C external/simulation-simple submodule update --init --recursive
~~~

已有目录先核对 origin，再复用 checkout。external/simulation-simple 由 .gitignore 忽略，仿真仓库维护自己的版本和 mujoco_menagerie 子模块。

如果尚未安装 uv，可使用其安装脚本；重新打开终端或将安装目录加入 PATH 后，再检查版本。

~~~bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
~~~

在支持 MuJoCo 渲染的机器上执行：

~~~bash
uv python install 3.11
uv sync --project external/simulation-simple --locked
uv pip install --python external/simulation-simple/.venv/bin/python --no-deps -e .

sim_python="$PWD/external/simulation-simple/.venv/bin/python"
"$sim_python" -c 'import torch, torchvision, av, pyarrow, mujoco; print(torch.__version__, torchvision.__version__, mujoco.__version__)'
"$sim_python" -m awbc_training.eval_sim --help
~~~

仿真环境需要能够导入 DiT 训练依赖。使用独立的已有环境时，将 sim_python 及评测配置的 python、simulation_root 改成实际位置。

Linux NVIDIA EGL 渲染需要驱动和 EGL/OpenGL 库。例如 Ubuntu：

~~~bash
sudo apt-get update
sudo apt-get install -y build-essential linux-libc-dev git curl ca-certificates   ffmpeg libegl1 libgl1 libopengl0

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
~~~

## 9. 独立评测策略成功率

在同一 shell 中设置 data、actor_run、sim_python 后执行：

~~~bash
"$sim_python" -m awbc_training.eval_sim \
  --checkpoint "$actor_run/checkpoints/latest_checkpoint.pt" \
  --weights ema --device cuda \
  --dataset-root "$data" \
  --mapping configs/eval_sim_block_insertion.json \
  --simulation-root external/simulation-simple \
  --seed 10000 --noise-seed 20000 --episodes 20 \
  --max-control-ticks 2500 \
  --output-dir "$actor_run/eval_20"
~~~

weights=raw 加载普通权重，ema 加载 EMA。每次评测用新输出目录。

添加 `--save-video` 可将每个正常结束的 episode（包括失败、截断和步数上限）保存为 `<output-dir>/videos/seed_<seed>.mp4`，对应 rollouts.jsonl 记录包含相对路径 `video_path`。`--video-camera` 默认为 `eye_rgb`，无效名称会在 rollout 前报错并列出可用相机。视频使用解析出的 action FPS，包含 reset 帧、每个动作保持周期后的画面和提前结束的最后一帧。推理或录像异常时清理未完成视频，错误仍写入 errors.jsonl。

单 episode 录像示例（路径均由运行时提供）：

~~~bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "<PYTHON_ENV>/bin/python" -m awbc_training.eval_sim \
  --checkpoint "<CHECKPOINT.pt>" --weights ema --device cuda \
  --dataset-root "<DATASET_ROOT>" --simulation-root "<SIMULATION_SIMPLE_ROOT>" \
  --mapping "<DIT_ROOT>/configs/eval_sim_block_insertion.json" \
  --episodes 1 --save-video --video-camera eye_rgb --output-dir "<NEW_OUTPUT_DIR>"
~~~

映射文件明确相机对应关系、原始状态顺序和动作语义，维度仅用于校验。默认适配本项目原始 44 维状态、16 维动作及单左臂控制。原始状态依次拼接 joint_pos、joint_vel、gripper_pos、ee_pos、ee_quat，各字段内部左臂在前。关节角用弧度、速度用弧度每秒，夹爪状态是驱动关节角；末端位置用米，四元数为 WXYZ。动作是绝对位置目标，关节用弧度，夹爪用 actuator coordinate。

默认相机为 eye、left_wrist、right_wrist 的 640×480 uint8 RGB。推理接收完整原始状态，由 checkpoint 切片并归一化，反归一化动作写入选定臂，另一臂保持 reset 后的控制目标。

动作频率来自 checkpoint 或 meta/info.json，也可显式 --action-fps 25。100 Hz 控制对 25 Hz 动作，每个动作保持 4 个 tick。每 tick 检查成功、终止、截断和预算。2500 tick 为最多 25 秒，1500 tick 为 15 秒。

输出 rollouts.jsonl、summary.json 和 errors.jsonl。成功取环境 info.success。异常时 complete=false、success_rate=null，保留已完成记录。当前本地评测验收使用假环境；真实 MuJoCo 闭环和任务成功率需在可渲染机器上确认。

## 10. 训练中验证成功率

在 policy JSON 增加 sim_eval。评测期间训练暂停，由独立 Python 子进程加载当前 step 的 actor 快照。

~~~json
{
  "sim_eval": {
    "enabled": true,
    "start_step": 10000,
    "update_interval": 5000,
    "weights": "ema",
    "python": "/absolute/path/to/simulation-simple/.venv/bin/python",
    "simulation_root": "/absolute/path/to/simulation-simple",
    "mapping": "/absolute/path/to/dit/configs/eval_sim_block_insertion.json",
    "device": "cuda",
    "seed": 10000,
    "noise_seed": 20000,
    "episodes": 20,
    "max_control_ticks": 2500,
    "timeout_seconds": 3600
  }
}
~~~

默认关闭仿真验证。以上配置在第 10000 次更新首次评测，然后为 15000、20000……。训练结束且达到 start_step 时补评最终 step，恢复后去重。低于开始步数的短训练不会启动仿真。普通离线 FM 验证单独由 validation 配置管理。

结果在 actor_run/sim_eval/step_XXXXXXXX/，包含 actor.pt、evaluation.json、stdout.log、stderr.log 和 results/。日志记录成功率、成功和完成条数、完整性及耗时。异常或超时后训练继续，该次成功率为 null。训练与评测共用 GPU 时，训练模型及优化器仍占用显存。

## 11. 离线推理

~~~python
from awbc_training.inference import load_policy

policy = load_policy("/path/to/actor.pt", use_ema=True, device="cpu")
# raw_images 是相机名到原始 uint8 RGB 图像的字典。
# raw_state 是完整原始状态向量，顺序与训练 metadata 一致。
raw_observation = {"images": raw_images, "state": raw_state}
observation = policy.prepare_observation(raw_observation)
prediction = policy.predict_action_chunk(observation)
execution = policy.predict_action(observation)
~~~

相机名使用 policy.config.camera_names。需要对比预测块与执行块时，两个调用传入同一份 initial_noise；执行块应等于完整预测块的前 n_action_steps 步。已由 dataset 切片、归一化的 batch 直接进入模型前向。

## 12. 常见问题与源码

| 现象 | 检查 |
|---|---|
| 找不到 awbc-train | 激活安装环境，或 python -m awbc_training.train |
| CUDA 为 False | 当前 PyTorch 构建、驱动、设备可见性 |
| Advantage 对齐失败 | critic 和 policy 的索引顺序、窗口、split、有效长度 |
| 导出找不到旧路径 | 用 export --set 覆盖数据与标签路径 |
| 缺少末帧进度 | 为当前实际末帧提供原始评分，复制帧显式沿用被复制帧评分 |
| 没触发仿真验证 | enabled、start_step 及训练实际完成的更新数 |
| success_rate 为 null | 对应 errors.jsonl、evaluation.json、stderr.log |
| 新实验目录已存在 | 更换输出目录，继续原实验使用 --resume |

src/awbc_data 负责轨迹、窗口和标签，src/awbc_models 负责模型、loss 和权重，src/awbc_training 负责训练、导出、推理与评测。

更多接口见 [数据说明](docs/DATA.md)、[模型结构](docs/ARCHITECTURE.md)、[训练说明](docs/TRAINING.md) 和 [数据审计](docs/data_audit/README.md)。
