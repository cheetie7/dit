# 10 步动作块 Dataset

这是独立于 LeRobot 策略实现的数据层，只使用 PyTorch Dataset、PyArrow、PyAV 和 Pillow，没有神经网络。数字列加载到内存，MP4 按需解码，适用于当前约 5 万帧的数据规模。

## 使用

从项目根目录运行，现有 `lerobot-awbc/.venv` 已具备依赖：

```bash
lerobot-awbc/.venv/bin/python -m awbc_data.inspect \
  --root block_insertion_real_combined_v30 \
  --output artifacts/awbc_data
```

在其他环境中安装 `awbc_data/requirements.txt`，然后使用该环境的 Python。验收输出包括：

- `batch_report.json`：随机 batch **全部张量**的形状、类型和范围。
- `batch.png`：随机 batch 前四个样本的三路图像及 8 维原始动作曲线。
- `tail.png`：额外抽取的不足 10 步尾部样本，灰色部分为 padding。
- `normalization.json`：仅训练轨迹的 min/max/mean/std/count。
- `split.json`：训练、验证 episode ID，可用于复现实验。

```python
from awbc_data import build_datasets
from torch.utils.data import DataLoader

train, val = build_datasets(
    "/root/autodl-fs/datasets/block_insertion_real_combined_v30_advantage",
    image_size=(240, 320),
    require_advantage=True,
)
loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
batch = next(iter(loader))
```

脚本使用多进程时，把构建 DataLoader 的入口放在 `if __name__ == "__main__":` 内。

## 返回值

下表为默认 DataLoader 拼接后的形状，单个样本没有 B 维。

| 字段 | 形状 | 说明 |
|---|---|---|
| `observation.images` | `[B, 3, 3, 240, 320]` | `[batch, camera, RGB, H, W]`，ImageNet 归一化 |
| `observation.state` | `[B, 8]` | 起点 state，训练 min/max 归一化 |
| `action` | `[B, 10, 8]` | 从当前帧开始的 10 个动作，训练 min/max 归一化 |
| `action_is_pad` | `[B, 10]` bool | **True 表示填充，应排除损失** |
| `advantage` | `[B]` | 原始 chunk 标签，保留负值 |
| `advantage_valid` | `[B]` bool | 原始数据为 False，标注版为 True |
| `episode_index`, `frame_index` | `[B]` int64 | 溯源起点 |
| `timestamp` | `[B]` float64 | episode 内观测时间 |
| `video_timestamp` | `[B, 3]` float64 | 各视频实际解码帧的 PTS 秒数 |
| `video_target_timestamp` | `[B, 3]` float64 | 通过 episode 元数据映射出的目标时间 |
| `valid_length` | `[B]` int64 | 真实动作数，1～10 |

相机顺序为 `train.source.cameras`，当前数据是 **top、left、right**。图像范围约为 `[-2.118, 2.640]`，归一化后不再是 `[0,1]`。

## 时间与统计约定

- 每个 episode 按 `frame_index` 排序，允许多个 episode 共用 Parquet，也允许一个 episode 跨数值文件。
- 样本起点为 `0,10,20,...`，观测使用 `s_t` 和同一时刻的图像；监督为 `a_t,...,a_(t+9)`，不向后错移一帧。
- 对 N 帧 episode 只使用 N−1 个 transition。终止行不作为动作监督。尾块重复最后一个真实动作至 10 步，用 mask 排除补位。
- 视频位置为 `videos/<camera>/from_timestamp + (timestamp[t] - timestamp[0])`，按 MP4 PTS 找最近帧，并检查误差不超过半帧加 1ms。不会把 episode 内帧号误当成共享视频的帧号。
- 图像解码为 RGB，Pillow bilinear resize 到 H=240、W=320，再转 float32、除以 255、按 ImageNet mean/std 归一化；无随机增强。
- 图像处理集中在 `ImagePreprocessor`，Dataset 与推理端复用同一实现。通过 `build_datasets(..., image_preprocess=ImagePreprocessConfig(size=(240,320), mode="letterbox", padding_rgb=(0,0,0)))` 可配置等比例缩放与居中 padding；默认 `mode="resize"`。不使用 Center Crop，编码器不再次归一化。显式预处理配置优先于旧的 `image_size` 参数。
- 默认按 episode ID 顺序把末尾 ceil(10%) 留作验证，与数据说明一致。`shuffle_episodes=True, seed=42` 可随机划分完整轨迹。
- state 统计使用训练轨迹全部真实状态，包括终止观测；action 统计只使用训练轨迹真实 transition 的动作，排除终止行与 padding。每行只统计一次。
- 数值默认 `2*(x-min)/(max-min)-1`。常量维度输出 0；验证集复用训练统计，不裁剪，因此可能超出 `[-1,1]`。可用 `train.denormalize("action", tensor)` 还原动作。
- 本地未标注数据的 advantage=0 仅为占位，`advantage_valid=False`。正式 AWBC 应使用 `require_advantage=True`，避免把缺失标签当作零 advantage。
- 标注版只从标注 chunk 起点取样，校验 10 步边界、有效长度与标签，不重复累计行级 advantage。不在 Dataset 内计算指数权重。

后续训练应先按 `~action_is_pad` 对每个 chunk 的动作损失求平均，再乘该 chunk 的 AWBC 权重。此阶段只交付数据。

## 验证

```bash
lerobot-awbc/.venv/bin/python -m pytest tests/test_awbc_data.py -q
```

测试使用实际生成的 MP4 和 Parquet，覆盖共享视频偏移、跨 Parquet 的 episode、图像内容与时间对齐、尾块填充、终止动作排除、训练/验证统计隔离、正负 advantage、多进程 spawn 读取。

本地原始数据验收得到 131/15 条训练/验证轨迹，4731/521 个 chunk。随机 batch B=4 的图像形状 `[4,3,3,240,320]`、state `[4,8]`、action `[4,10,8]`；完整范围记录在输出 JSON 中。
