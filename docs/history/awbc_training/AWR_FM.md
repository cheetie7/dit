# 阶段 8：Advantage-Weighted Flow Matching

## 当前状态与范围

通用 Precomputed Advantage 接口、严格窗口对齐、训练集 Advantage Normalization、权重转换、AWR-FM Trainer 和诊断图已完成。FM baseline 保留。新实验默认 action_horizon=10，与已评好的 RoboMeter chunk 一致；执行长度仍独立为 8。

真实采样频率是 30 Hz，10 个 transition 约为 0.333 秒，不是提示词中的 20 Hz。代码从 Dataset metadata 检查频率，不自动重采样或换算 horizon。

本地两份原始数据尚无 annotation 列；用户表示会下载标注数据。本轮只使用清楚标为 synthetic 的观测、动作与 Advantage 做训练测试，没有启动真实数据 AWR-FM，也没有伪造 RoboMeter 标签。真实标签的数量、分布、模型版本和来源偏差待下载后审查。

## 文件

新增：

- `awbc_data/advantage.py`：AdvantageProvider、PrecomputedAdvantageProvider、RewardToAdvantageAdapter 接口、窗口 metadata、Dataset 包装。
- `awbc_models/advantage_weighting.py`：独立的 AdvantageNormalizer 和 AdvantageWeighting。
- `awbc_training/advantage.py`：数据绑定、训练集统计拟合/加载与现有 Policy 的薄调用层。
- `awbc_training/advantage_analysis.py`：统计、ESS、按轨迹权重质量和分布图。
- `awbc_training/inspect_advantage.py`：合成观测/标签的有界 FM/AWR 对照诊断。
- `awbc_training/configs/fm_h10.json`、`robometer_awr_fm_h10.json`：公平对照配置。
- `awbc_training/configs/robometer_advantage_metadata.template.json`：真实来源记录模板。
- `awbc_training/requirements.txt`：增加诊断图需要的 Matplotlib。
- `tests/test_advantage_weighting.py` 与本说明。

修改：训练配置、Trainer、train 入口、日志字段、batch 元数据搬运、模块导出；DiT 默认 horizon 和两个基础 JSON 配置改为 10。Policy 配置新增 from_dict，以便旧 checkpoint 使用当时保存的真实 horizon 恢复；模型前向没有重写。

ResNet、Observation Encoder、Flow Matching Objective、Euler 及原 Dataset 核心实现不变。原有 FM checkpoint 和实验配置文件未覆盖。

## Advantage 数据接口

每行标准记录：

```json
{
  "sample_id": "ep000003_t000020_h10",
  "trajectory_id": 3,
  "start_timestep": 20,
  "end_timestep": 30,
  "action_horizon": 10,
  "advantage": 0.02,
  "advantage_source": "robometer",
  "advantage_model_version": "实际模型版本"
}
```

end_timestep 是真实动作范围的右开端点。例如尾块起点 380、有效 3 步，则 end=383，而 nominal horizon 仍为 10。字段是帧索引，不是秒。trajectory_id 对应当前 LeRobot 原始 episode_index。

Provider 声明输出语义与来源，接口为 get_advantage(sample_metadata)。当前实现预计算表，不调用新的评分网络。标准表支持 Parquet、CSV、JSON 列表和 JSONL；不接受没有窗口坐标证明的裸数组对齐。

同时支持原 RoboMeter 导出：

1. 标注版 LeRobot 的 annotation.advantage，只在 action_chunk_start 且 valid 的起点取一次。
2. 已有 labeling/chunk_advantages.jsonl，直接读取其中算好的 advantage，转换字段名，不重新生成评分或差分公式。

同一个 chunk 的 Advantage 在真实动作行重复存储时，不会把它们相加，也不会变成多个训练样本。

## 严格对齐与缺失策略

实际 lookup key 为 `(trajectory_id, start_timestep, action_horizon)`。构造 Provider 时检查重复 sample_id、重复窗口键、有限 Advantage、来源版本与表声明；在绑定前校验完整轨迹窗口集合、数量、fps 和每个真实 end_timestep。声明评价范围时，标签不能超出该轨迹范围。

与当前 Dataset 的原始窗口逐一 join 后，AdvantageDataset 才作为 DataLoader 输入。Shuffle 只打乱已经绑定好的样本，绝不按 batch 行号查标签。sample_id、轨迹/起点/终点/horizon、source/version 随样本返回。

默认缺失标签即报错。可显式配置 missing_label_policy="drop"；删除数量和保留的训练样本数进入 provenance/manifest，不赋零。没有任何标签可训练时明确报错。Validation 数据不因缺标签而删减，始终保留原固定验证窗口。

32 步/16 步标签不能套在 10 步窗口上；key、数量或 end 检查会在训练前失败，不复制、不按比例拆分、不自动改变 Policy horizon。

## 来源与 RoboMeter 语义

需要 sidecar metadata，模板中尚未知的字段保持 null，provenance_reviewed=false。真实入口要求提供实际 model_version 和已完成的来源审查标记，避免把没有依据的评分文件当成正式实验标签。

记录 source、semantics、model/version、checkpoint hash、生成日期、Advantage 定义、名义窗口、fps、评分模型训练轨迹范围和被评价数据范围。Provider 还从实际标签键输出 labeled_trajectory_ids、表 fingerprint 和与 Policy train/val 的已知重叠；训练范围未知时报告 null，不声称无重叠。training_trajectory_ids 应使用当前数据集的原始 ID 空间，外部语料不能仅凭相同数字 ID 推断重叠。

检查已有 export_advantage_dataset.py 后，当前已知定义是 progress_increment_proxy：每两个 10-transition chunk 共享边界进度差，各分到 delta/2；单个尾 chunk 使用完整 delta；保留负值，终止进度强制为 1。它是已有研究中的进度增量代理指标，不是 Q−V，也不是未经转换的 raw progress。

本阶段只消费已经算好的 annotation.advantage/JSONL advantage，不重新执行该分配。真实文件下载后仍需核对本批标签是否由这个脚本/定义产生。如果只是 20/32 步整体分数，没有已明确生成的 10 步 Advantage，不会临时按比例拆给新窗口。

原始 reward/progress/quality semantics 会被拒绝；RewardToAdvantageAdapter 只是抽象接口，没有自行规定转换公式。STEAM、在线 RoboMeter 或 IQL 后续可实现同一 Provider 协议；当前不增加没有模型依据的空壳网络或假评分器。

## Advantage Normalization

模式为 none、center、zscore。均值与总体标准差只从完整训练轨迹的唯一窗口标签拟合，先于 smoke 子集截取；Validation 标签不参与。

```text
none:   A' = A
center: A' = A - mean
zscore: A' = (A - mean)/(std + 1e-8)
```

常量标签 std=0 时仍有限。保存 advantage_stats.json，包含 mean/std/count、训练轨迹 IDs、标签 fingerprint 和 sample ID hash。可通过 statistics_path 加载同一数据版本/训练划分的统计。Checkpoint 保存 normalizer buffers，resume 恢复它们；Validation 不重新拟合，也不调用 weighting。

已经标准化的输入可以选 none，避免二次标准化。

## 权重与 Loss

```python
weighting = AdvantageWeighting(
    normalizer,
    beta=1.0,
    min_weight=0.1,
    max_weight=10.0,
    detach_weights=True,
)
sample_weight = weighting(batch["advantage"])  # [B]
```

先计算 `log_w=A'/beta`，再在配置的 log(min/max) 范围裁剪，最后 exp。beta 必须有限且为正，输出必须有限且大于零。min/max 可分别设置 null；同时为 null 即关闭 clipping，无法表示的 exp 会明确报错，不静默生成 Inf/零权重。默认在权重入口 detach Advantage；模块也允许配置保留 tensor 梯度，但当前训练只用固定预计算标签，不训练评分模型。

调用复用阶段 5/6：

```python
output = policy(batch, sample_weight=sample_weight)
weighted_loss = output["loss"]
```

实际加权 MSE 仍由原 FlowMatchingObjective 计算：先排除 action padding，按每个样本真实动作元素数平均，再做 `sum(w_i*L_i)/sum(w_i)`。没有第二套 loss。训练薄层 training_forward 增加 weighted_loss（loss 的别名）、unweighted_loss=per_sample_loss.mean() 和原始 advantage，保留 per_sample_loss/sample_weight。

当权重全部相同，原 Objective 的常量分支保证与普通均值一致。零权重/非法权重和全 padding 的处理沿用已通过测试的阶段 5 规则；AdvantageWeighting 本身只产生正权重。

## Trainer 与恢复

training.objective 支持 fm / awr_fm；AWR 还需 advantage_weighting.enabled=true。FM 路径不加载标签、不生成权重。没有 WeightedRandomSampler，继续使用均匀 shuffle 和 loss weighting。

梯度累积也按整个更新窗口归一：假设某 minibatch 的权重和为 S_m，整个累积窗口权重和为 S，则对该 minibatch 的归一化 loss 乘 S_m/S。这样等价于对有效大 batch 做一次加权 loss。直接平均每个 minibatch 的归一化加权 loss 会产生偏差，因此没有沿用 FM 的样本数比例来缩放 AWR loss。

累积窗口的权重和由已经按窗口键绑定的 Dataset 标签预计算；实际传给 Policy 的权重仍由当前 batch 自带 Advantage 生成，不按 batch 行号取标签。统计量/温度固定，无动态 Advantage 网络。

Checkpoint 增加 weighting buffers、部分 epoch 的原始标签/权重/loss 统计。manifest 记录 provider fingerprint。恢复不允许悄悄改变 objective、标签版本或权重配置。旧 FM checkpoint 用其保存的 policy_config 构建网络，因此旧 16 步实验不会被新的 10 步默认值误加载；不能把一个 16 步 checkpoint 静默裁成 10 步。

Validation 直接调用 Policy，不经过 training_forward/weighting，继续使用阶段 7 的固定验证顺序、固定 noise/time 和原始/EMA 普通 FM loss；不更新 normalizer、Optimizer 或 EMA。

## 统计分析

每次更新窗口和每个 epoch 记录 Advantage mean/std/min/max/median、weight mean/std/min/max、clip_fraction、minimum_fraction、ESS 和 ESS ratio。ESS 为 `(sum(w))²/sum(w²)`；更新级 ratio 除以当前累积窗口样本数，epoch 级 ratio 除以已处理的 epoch 样本数。

同时记录每条轨迹的权重质量占比和最大轨迹占比，便于发现权重集中。保存原始 unweighted loss、weighted loss 与原始/EMA validation loss，可直接比较训练加权下降时验证是否恶化。

输出 advantage_analysis/epoch_*_step_*/analysis.json 和 distributions.png，包含两种直方图、Advantage–Weight 散点、分位数样本量及平均未加权训练 loss。相同 Advantage 不被任意拆入不同分位桶。图中 loss 来自训练过程，不应当据此宣称标签质量或因果效果。

## 对比配置与真实数据接入

- Exp A：configs/fm_h10.json，普通 FM。
- Exp B：configs/robometer_awr_fm_h10.json，RoboMeter 预计算标签的 AWR-FM。
- STEAM：可沿用相同预计算协议，但当前没有已核实的 10 步标签或模型定义，因此不提供虚构的可运行评分网络。

两个正式配置保持相同 seed=42、轨迹划分、模型、10 步 horizon、batch、训练预算、验证条件和初始化策略，只改变 objective/标签入口。RoboMeter 标注版应保留原始图像/state/action；下载后会核对其范围及来源。

真实入口示例（本轮没有执行）：

```bash
python -m awbc_training.train --config awbc_training/configs/robometer_awr_fm_h10.json \
  --set dataset.root=/path/to/scored_dataset \
  --set advantage_weighting.metadata_path=/path/to/reviewed_metadata.json
```

如果只下载 chunk_advantages.jsonl，可使用原始 Dataset 加表：

```bash
python -m awbc_training.train --config awbc_training/configs/robometer_awr_fm_h10.json \
  --set dataset.root=block_insertion_real_combined_v30 \
  --set advantage_weighting.labels_path=/path/to/chunk_advantages.jsonl \
  --set advantage_weighting.metadata_path=/path/to/reviewed_metadata.json
```

新增 Matplotlib 依赖列于 awbc_training/requirements.txt。日志仍可关闭 WandB，只使用本地记录。未增加 best checkpoint，未覆盖旧 FM 实验。

## 测试与合成验收

```bash
MPLCONFIGDIR=/tmp/awbc-matplotlib lerobot-awbc/.venv/bin/python -m pytest tests/test_advantage_weighting.py -q

MPLCONFIGDIR=/tmp/awbc-matplotlib lerobot-awbc/.venv/bin/python -m pytest \
  tests/test_advantage_weighting.py tests/test_training.py tests/test_flow_matching_policy.py \
  tests/test_flow_matching.py tests/test_awbc_dit.py tests/test_awbc_observation.py \
  tests/test_awbc_vision.py tests/test_image_preprocessing.py tests/test_awbc_data.py -q

# 只生成合成观测和合成标签，FM/AWR 各 3 次更新
MPLCONFIGDIR=/tmp/awbc-matplotlib lerobot-awbc/.venv/bin/python -m awbc_training.inspect_advantage
```

23 项新增测试、116 项联合回归通过（10.81 秒）。覆盖标签格式/重复/缺失/错误轨迹/错误起点/分数索引/错误 horizon/end/fps、shuffle 绑定、RoboMeter 旧表及标注 Dataset 字段适配、raw score 拒绝、train-only normalization 与加载、None/Center/Z-score、温度/裁剪/有限值/detach、等权等价、14/6 手算加权、padding、完整视觉/状态/DiT 梯度、两种模式更新与保存、未加权验证、加权累积的大 batch 等价、AWR 中途恢复与统计延续、旧 horizon 配置恢复。旧 DataLoader spawn 测试需要 PyTorch 共享内存管理进程，在允许启动该进程的环境中通过；未修改其读取逻辑。

诊断目录：`outputs/stage8_synthetic-20261030-221203-972421`。Synthetic FM 和 AWR 都完成 3 次更新、普通/EMA 验证及 checkpoint 保存，输入使用人工小图和动作，不是机器人原始数据。完整真实架构的各组件被复用，诊断隐藏维度为 32、每类 Transformer 为 1 层以控制规模，未下载或训练新评分模型。

合成训练的 6 个 Advantage 为 [-2,-1,0,1,2,3]，训练统计 mean=0.5、std=1.707825；验证标签为另一组大值，不参与拟合。示例 beta=0.5、权重裁剪 [0.1,10]：

| 指标 | 合成结果 |
|---|---:|
| Weight mean / std | 3.069703 / 3.665007 |
| Weight min / max | 约 0.1 / 10 |
| Clip fraction | 33.33% |
| ESS | 2.473751 / 6 |
| ESS Ratio | 0.412292 |
| 最大轨迹权重质量占比 | 95.50% |

这个高集中度只演示诊断能发现问题，不是实际 RoboMeter 的分布或效果。完整结果在 artifacts/awbc_awr/synthetic_report.json。

## 待真实文件确认的事项

1. 标注版 Dataset 或实际 chunk_advantages.jsonl 的本地路径与完整窗口覆盖。
2. 本次 RoboMeter 的 model/checkpoint 版本、hash、生成日期及是否确实采用已有进度增量定义。
3. 评分模型训练数据与当前 146 条轨迹的重叠；未知时明确记为未知，不能宣称 held-out。
4. 现有相邻两 chunk 共享 delta/2 的相关性、终止进度强制 1 和成功轨迹假设是否适合此次 AWR-FM 研究。
5. beta、归一化和裁剪对 ESS 的影响，需要在相同 10 步 FM baseline 上做真实对照，不能从合成 smoke 推断收益。
6. 若采用 cross-fitting，还需每折模型版本与训练范围账本；当前 Provider 是单一声明版本的预计算数据源。
7. STEAM 需要官方实现/权重、输入与 H 的精确定义，以及经过确认的 10 步指标转换。当前没有这些材料，不生成假的 Adapter 或 Advantage 公式。

真实 AWR-FM 验收等待用户提供已评数据；没有启动完整训练。
