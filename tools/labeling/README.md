# 已有 RoboMeter 标注工具

保留原有 manifest → boundary inference → audited export → visualization 工具，便于追溯当前标签定义。它们不是训练时依赖，项目整理时没有运行评分或重新生成标签。

- prepare_advantage_manifest.py：产生边界/动作块 manifest。
- run_robometer_boundary_inference.py：调用外部 RoboMeter 官方实现与权重，生成边界进度并分配 chunk 指标。
- export_advantage_dataset.py：审计后导出独立标注版 LeRobot v3 数据集。
- visualize_robometer_progress.py：进度视频可视化。

各脚本使用 `python tools/labeling/<script>.py --help` 查看参数。评分脚本需要另外准备兼容的 RoboMeter 包、decord、官方模型权重；可视化需要 imageio-ffmpeg。它们没有作为项目训练安装的强制依赖。当前包不包含评分模型、权重、虚构的评分结构或机器人 SDK。

现有公式、强制终点及未解决的来源信息见 ../../docs/data_audit/README.md。不要把 raw progress 当作独立的 Q−V。
