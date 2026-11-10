# 数据与标签

当前支持 LeRobot v3 的 Parquet 数值数据和带时间戳 MP4。每条轨迹的末帧是终点观测，没有已观测 next state，因此动作使用前 length−1 个 transition。训练从轨迹起点按 `policy.dit.action_horizon` 无重叠切块，默认 10 步。窗口以轨迹为界，尾块重复最后一个有效动作填充，`action_is_pad=True` 的位置不参与损失。

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

Batch 图像 `[B,N_cam,3,240,320]`，state `[B,8]`，动作 `[B,H,8]`，mask `[B,H]`，其中 `H` 是动作窗口长度。RGB 保持通道顺序，Resize 或 letterbox 配置化，不做 Center Crop，使用 ImageNet mean/std，Dataset 内只归一化一次。推理复用同一个 ImagePreprocessor。

状态/动作 min-max 统计只从训练轨迹计算；验证与推理复用。同一维 min=max 时归一化为 0。数据自带的全局 meta/stats.json 不直接作为训练统计。Advantage 统计按训练集唯一窗口计算，不按重复动作行计数，支持 none/center/zscore。

AWR 标签按 `(trajectory_id,start_timestep,action_horizon)` 匹配。`end_timestep` 是真实有效动作的右开端点，尾块的名义长度仍为 `H`。Provider 检查唯一性、覆盖、horizon、fps、真实 end 与来源声明。默认缺失即报错，显式 drop 时只过滤训练窗口，验证集仍保持固定。

已评数据可以从 Parquet annotation 或外部 JSON/JSONL/CSV/Parquet 表读取，不必安装 RoboMeter。原始 reward/progress/quality score 不自动转换成 Advantage。当前 RoboMeter 指标是已有导出流程生成的进度增量 proxy：两块共享边界差分各取一半，单独尾块取完整差分，终点进度强制为 1，负值保留。

用 10 步 RoboMeter 进度增量标签训练 `H=20、30、40…` 时，先校验基础标签，再把同轨迹窗口内连续基础块的标量相加，每块取一次。尾窗口只合计实际覆盖的基础块。长窗口可能只覆盖某个评分组的一部分，此时采用原导出中分配给该块的值。合成标签先在训练集上计算归一化统计，再用于 AWR 加权。统计量与合成标签指纹绑定，来源记录包含基础标签指纹、基础块长度和新窗口步幅。验证仍用普通 FM loss。

来源说明优先使用显式 metadata_path，其次是 labeling/advantage_metadata.json。已核验的旧 RoboMeter 导出只有 summary.json 与 chunk_advantages.jsonl 时，入口支持直接加载：检查表内已存 Advantage 的公式、分组与 horizon，再验证窗口覆盖和 JSONL/Parquet 数值一致性。缺失来源字段保存为 null，输出来源不完整提示并继续使用已经算好的标签；不伪造模型版本、不把 provenance_reviewed 自动设为 true。它不是将任意评分转换成 Advantage 的通用兜底。

2026-11-01 已使用当前真实数据验证：5,252 块完整覆盖，train/val=4,758/494，一个完整块与一个 7 步尾块组成 batch，AWR-FM 前向及 backward 通过，梯度有限且冻结 ResNet 无梯度；未执行 optimizer update。来源记录中的精确模型版本与评分模型训练范围仍未知。

有关来源未知项、实际分布及尾块检查，见 [数据审计](data_audit/README.md)。
