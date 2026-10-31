# 下载的 RoboMeter 标注数据检查

数据目录：`/Users/cheetie/Downloads/block_insertion_real_combined_v30_advantage`。

这是包含 chunk 进度增量标签的 LeRobot v3 数据集，能被现有 Dataset 和 PrecomputedAdvantageProvider 读取。本次仅检查数据与计算诊断，没有启动训练、修改标签或标记来源审查完成。

| 项目 | 实际结果 |
|---|---:|
| 轨迹 / 帧 / 有效 transition | 146 / 51,996 / 51,850 |
| 采样频率 / 名义 chunk | 30 Hz / 10 步 |
| chunk / 不足 10 步尾块 | 5,252 / 133 |
| 负 / 正标签 | 1,694 / 3,558 |
| 标签 min / max | -0.142837 / 0.821792 |
| 标签 mean / std | 0.022569 / 0.054475 |
| 按轨迹 seed=42 划分 | train 131 条、4,758 块；val 15 条、494 块 |

检查了所有窗口的轨迹、起止帧、10 步 horizon、实际有效长度、标签重复与缺失、JSONL/Parquet 标签一致性，以及进度边界表与标签计算公式。全部通过，未跨轨迹。所有原有数值列与本地原始数据逐列一致；4 个 MP4 的 SHA256 与导出记录一致。

## 标签的具体含义

`boundary_progress.jsonl` 保存边界处 RoboMeter 的原始及实际使用进度。`chunk_advantages.jsonl` 和 Parquet `annotation.*` 保存已经分配好的 chunk 标签。

通常每 20 个 transition 为一组，拆成相邻两个 10 步 chunk，两者共享：

```text
delta = progress(group_end) - progress(group_start)
advantage(chunk_1) = advantage(chunk_2) = delta / 2
```

单独一个尾 chunk 使用完整 delta。5,174 个 chunk 的 divisor=2，78 个 divisor=1。这里的名义训练块确实是 10 步，但绝大多数不是独立以各自 10 步起止进度做差。

所有 146 条轨迹的终点进度被强制设为 1，负进度增量保留。此值的语义是 `progress_increment_proxy`，不能把它解释成原始 progress 或已经估计好的 Q−V。每个真实动作行重复同一 chunk 标签，读取时只取 chunk 起点一次。

例如 episode 0 的 frame 0 和 frame 10 两个块都得到 -0.001418889，它们共用 frame 0→20 的进度变化。最大标签 0.821792 来自 episode 103 的最后 7 步块；其原始终点进度约 0.300536，实际用于计算的终点是强制后的 1。

## 真实读取与 Mask

抽取完整块与有效 2 步的尾块，得到：

```text
images:        [2, 3, 3, 240, 320]   # top / left / right
state:         [2, 8]
action:        [2, 10, 8]
action_is_pad: [2, 10]
advantage:     [2]
valid_length:  [10, 2]
```

尾块后 8 个位置为 True。将这些位置预测误差设为 1e6、真实位置误差设为 1，现有 Objective 得到 per_sample_loss=[1,1]，加权 loss=1，确认 padding 被忽略。可视化见 `batch.png`；灰色动作点为 padding。图像时间对齐最大误差约 3.18e-7 秒。

状态、动作统计仅使用训练轨迹。Advantage 训练窗口统计 mean=0.02235357、std=0.05307252，未使用验证标签。未直接使用导出的全数据 stats.json 作为训练归一化统计。

## 权重诊断（没有训练）

按当前示例配置 zscore、beta=1、clip=[0.1,10]：

- Weight mean=1.422916，std=1.934149，median=0.757992。
- Clip fraction=3.45%。
- ESS=1,670.85 / 4,758，ESS ratio=35.12%。这是整个训练集的权重集中度，不是单个 batch 的 ESS。
- 涉及强制终点的组占训练窗口 4.08%，占总权重 11.19%。终点处理对权重分布有明显影响，不能把该诊断当作策略性能提升证据。

## 来源仍未知的部分

`labeling/summary.json` 明确记录模型路径 `/root/autodl-tmp/models/Robometer-4B`，但没有精确 checkpoint revision/hash、标签生成日期及评分模型训练轨迹范围。因此暂时无法判断评分模型是否见过当前轨迹，也不能声称是 held-out 评分。导出数据未包含训练入口所需的完整来源 sidecar；`observed_provenance.json` 只记录已查到的事实，未知字段保留 null，provenance_reviewed=false。

完整机器可读结果在 `report.json`。复现：

```bash
python tools/audit_dataset.py --root /path/to/scored_dataset --original /path/to/raw_dataset
```
