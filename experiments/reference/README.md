# 已发表口径的历史仿真参考源

amr-f62c064.zip 是项目提交 f62c0641bd0d56c86ddcf21d06e725c75d5b6a0f 的 amr_sched 目录原样归档。SHA-256：a9f249557b742f339331ccd781d8ed66d37fcc6be14c88d6d20020c158a0f164。归档命令：git archive --format=zip --output=amr-f62c064.zip f62c0641bd0d56c86ddcf21d06e725c75d5b6a0f amr_sched。

此归档用于复算既有结果表抽样和两条旧容量数值断言。辅助入口 tools/research_baseline.py 使用单独的 Python 进程，并核对归档摘要及加载路径。旧 CSV、指标断言、源版本和失败证据均保留；历史仿真不用于运行当前调度、ROS 2 或车辆控制。这里的抽样复算不代表历史全部 CSV 已重测，也不为旧版的共享通道容量问题提供通过结论。
