# 项目整理与验证

新项目的源码入口为 src/，使用 pyproject.toml 安装。三个原始模块名保持为 awbc_data、awbc_models、awbc_training，checkpoint state_dict 与保存的 Policy 配置继续可用。项目外旧源文件与实验结果保持原样，后续开发以本项目为准。

本次整理没有修改 Dataset 动作语义、ResNet、Observation Transformer、Conditional DiT、Flow Matching Objective、Euler 或 Trainer 的核心实现；原文件与迁入文件逐字节比较通过，SHA256 记录在 validation.json。

必要的可移植性调整：

1. 默认模型配置随 wheel 安装，消除对调用目录中 awbc_models/configs 的依赖。
2. 四个诊断工具按安装模块的位置计算源文件 hash。
3. CLI 默认配置使用 dataclass，示例实验统一放在 configs/，不依赖原工作区。
4. CLI 记录数据/模型/训练三个包的源代码 hash。
5. Resume 用 checkpoint 内的视觉骨干权重，跳过没有用途的预训练初始化下载。
6. 新增 load_policy，直接使用 checkpoint 中的模型配置、状态/动作统计和原始/EMA 权重，支持数据移动到另一台机器后推理。

## 验证结果

- 119 项测试通过，11.42 秒；包含旧 116 项回归与 3 项安装配置/推理恢复检查。
- 完整测试包含 DataLoader spawn worker，需要允许 PyTorch 共享内存管理进程；验证环境中通过。
- wheel 安装到工作区以外的独立目录，依赖使用本机已有环境；在 /tmp 运行入口与导入，模块实际来自 wheel 安装目录。没有在全新机器重新下载安装全部依赖。
- 安装包的合成 FM/AWR-FM 各完成 3 次更新、原始/EMA 验证、checkpoint 保存；见 synthetic_validation.json。
- 真实数据 FM：使用实际三相机预训练骨干和默认 256 维模型，4 个训练窗口、2 个验证窗口；先更新 1 次，保存，再通过 wheel 的 CLI 从 checkpoint 恢复到第 2 次。Optimizer 状态、scheduler=2、EMA=2 连续。
- 从恢复后的 checkpoint 加载 EMA，用真实观测进行 Euler 推理，输出 [1,10,8]，数值全部有限。
- Resume 与推理使用空的 TORCH_HOME 目录，未下载 ImageNet 权重。
- 真实数据仅执行上述两次普通 FM 更新，没有启动完整训练或真实 AWR-FM。真实 RoboMeter 的未知版本/训练范围继续明确保留。
- 硬件验证为 CPU/float32；精度逻辑的 CPU float32/BF16/FP16 单元测试通过，未实测 CUDA 显存或吞吐。

真实 smoke 输出留在本机 outputs/project_smoke，不进入源码压缩包。原始实验没有覆盖。机器可读结果见 validation.json。

## 交付格式

- 项目目录：含源码、配置、文档、测试与标注工具。
- 源码 zip：排除数据、outputs、权重、缓存、虚拟环境、本机 local 配置与构建中间文件。
- dist/ 中的 wheel / sdist：可安装分发；wheel 只包含运行包和默认模型配置，源码分发另含测试/文档/工具/实验配置。

旧阶段文档归档在 history/，保留当时的验收数据，其中旧路径、16 步实验与虚拟环境命令不作为当前项目运行指引。当前以顶层 README、configs 和 DATA/ARCHITECTURE/TRAINING 文档为准。
