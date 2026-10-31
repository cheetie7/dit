# 数据与标签

当前支持 LeRobot v3 的 Parquet 数值数据和带时间戳 MP4。每条轨迹的末帧是终点观测，没有已观测 next state，因此动作使用前 length−1 个 transition。从轨迹起点按 10 步无重叠切块，不跨轨迹；尾块重复最后一个有效动作填充，`action_is_pad=True` 的位置不参与损失。

当前数据为 146 条轨迹、51,996 帧、5,252 块、30 Hz、三个摄像头 top/left/right、state/action 均为 8 维。其中 133 个尾块不足 10 步。默认 seed=42，以完整轨迹按 tasks 分组划分，train 131 条/4,758 块，val 15 条/494 块。

```python
{
    "observation": {"images": ..., "state": ...},
    "action": ...,
    "action_is_pad": ...,
    "advantage": ...,
    "advantage_valid": ...,
    "episode_index": ...,
    "frame_index": ...,
    "valid_length": ...,
}
```

Batch 图像 `[B,N_cam,3,240,320]`，state `[B,8]`，动作 `[B,10,8]`，mask `[B,10]`。RGB 保持通道顺序，Resize 或 letterbox 配置化，不做 Center Crop，使用 ImageNet mean/std，Dataset 内只归一化一次。推理复用同一个 ImagePreprocessor。

状态/动作 min-max 统计只从训练轨迹计算；验证与推理复用。同一维 min=max 时归一化为 0。数据自带的全局 meta/stats.json 不直接作为训练统计。Advantage 统计按训练集唯一窗口计算，不按重复动作行计数，支持 none/center/zscore。

AWR 标签按 `(trajectory_id,start_timestep,action_horizon)` join。end_timestep 是真实有效动作的右开端点，尾块 nominal horizon 仍为 10。Provider 检查唯一性、覆盖、horizon、fps、真实 end 与来源声明；默认缺失即报错，显式 drop 时只过滤训练窗口，验证集仍保持固定。

已评数据可以从 Parquet annotation 或外部 JSON/JSONL/CSV/Parquet 表读取，不必安装 RoboMeter。原始 reward/progress/quality score 不自动转换成 Advantage。当前 RoboMeter 指标是已有导出流程生成的进度增量 proxy：两块共享边界差分各取一半，单独尾块取完整差分，终点进度强制为 1，负值保留。

有关来源未知项、实际分布及尾块检查，见 [数据审计](data_audit/README.md)。
