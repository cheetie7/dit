# 模型与损失

```text
RGB cameras [B,N_cam,3,H,W]
  → shared ImageNet ResNet-18 + FrozenBatchNorm, layer4 [B,512,Hf,Wf]
  → shared 1×1 projection to 256
  → spatial tokens + separate 2D sine positions / camera embedding
  → concatenate cameras in config order
state [B,8] → linear state token
  → Observation Transformer (state token + visual tokens)
  → observation_memory / observation_pos / observation_padding_mask
noise + normalized action + continuous timestep
  → Conditional DiT with observation cross-attention
  → predicted velocity [B,10,8]
```

没有全局池化或 Spatial Softmax。视觉 backbone 可选 frozen/layer4/all；冻结参数不进优化器。默认隐藏维度 256，Observation Encoder 两层，DiT 四层，均为 8 heads。相机共享 backbone/projection，逐图提取后按相机序列拼接；位置和特征分别保留。

Flow Matching 路径为 `x_t=(1-t)*noise+t*action`，t=0 是噪声，t=1 是动作，目标速度 `action-noise`。MSE 在 float32 聚合，按每个样本有效动作元素数归一化。全 padding 样本明确报错。

FM：`loss = per_sample_loss.mean()`。

AWR-FM：训练集 Advantage 归一化后，`log_w=A'/beta`，可在 log space 裁剪，然后 `w=exp(log_w)`，默认 detach。`loss=sum(w*per_sample_loss)/sum(w)`。保留 per_sample_loss、unweighted_loss、weighted_loss；权重只对应样本，不展开到 H×A。验证始终普通 FM。

Euler 从高斯噪声 t=0 向动作 t=1 积分；一次观测只编码一次。动作先在归一化空间生成，最后按训练统计反归一化。预测 horizon=10，执行步数 n_action_steps=8 独立配置。

核心模块从已验收版本迁入，没有重写其网络结构或数学目标。详细历史接口记录保存在 history/，其中旧命令与旧 16 步结果仅作历史参考。
