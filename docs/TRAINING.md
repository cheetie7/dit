# 训练与实验

入口是 `awbc-train`、`python train.py` 或 `python -m awbc_training.train`。三者使用同一 main。JSON 配置与点号 `--set key=value` 复用已有系统；`--set` 可以多次提供，布尔/null/数字按 JSON 解析。

流程：完整轨迹划分 → 训练集归一化统计 → 可选窗口标签绑定与 Advantage 统计 → Policy → AdamW 参数分组 → AMP → 前向与累积 → unscale/clip → optimizer → scheduler/EMA → 固定条件验证 → checkpoint 与日志。

冻结参数不进 AdamW，bias/norm 可关闭 weight decay，backbone 使用独立较小 LR。Warmup/cosine 的总步数为 optimizer update 次数；FP16 overflow 跳过更新时 scheduler、EMA 与 global_step 同步跳过。

FM 梯度累积按窗口内实际样本数归一化，支持末尾不完整窗口。AWR 按整个累积窗口的总权重归一化，避免把不同 minibatch 的加权均值简单平均。只有 optimizer 真正更新后才更新 scheduler 与 EMA。

CPU/CUDA 支持 float32、BF16 与带 GradScaler 的 FP16；MPS 按 PyTorch/系统能力检查。模型参数保持 float32，通过 autocast 运行前向。GPU 显存、裁剪前梯度范数、各参数组 LR、耗时及跳过更新计入日志。

Validation 使用 eval + inference_mode、固定 seed/顺序/batch size，原始和 EMA 模型共用相同 noise/timestep；不调用 Advantage Weighting，不拟合 normalizer，不更新参数或 EMA。

输出：

```text
outputs/<experiment>-<timestamp>/
  config.json
  policy_config.json
  normalization_stats.json
  split.json
  metadata.json
  optimizer.json
  batch_report.json
  train.log / metrics.csv / metrics.jsonl
  advantage_stats.json / advantage_provenance.json   # AWR
  advantage_analysis/                              # AWR 分布图与逐窗口诊断
  checkpoints/
    latest_checkpoint.pt
    epoch_XXXX.pt
```

Checkpoint 临时写入后原子替换，保留最近 N 个定期文件，不保存 best。恢复全部训练状态及数据位置；AWR 同时恢复 normalizer buffers 和未完成 epoch 的诊断统计。已有 checkpoint 模块名保持兼容。外部数据需保持可访问，换机器时可以覆盖 dataset.root；变更数据内容/划分/统计会拒绝作为同一次实验恢复。

每个 epoch 输出原始 Advantage 与权重统计、clip fraction、ESS/ratio、按轨迹权重占比、直方图、散点图与分位数 loss。WandB 是可选依赖，失败时继续本地日志。第一版不使用 WeightedRandomSampler，不计算新的 Advantage，不包含 IQL、STEAM 在线评分或机器人控制。

`configs/overfit.json` 是固定 batch 的 24 次更新诊断；`configs/smoke.json` 是 3 次更新诊断。`configs/fm.json` 与 `awr_fm.json` 是相同模型、划分、seed、batch/LR 的基线对照模板，参数不是最优值声明。更改 objective 请新建实验，不在同一个 checkpoint 上改变研究设置。
