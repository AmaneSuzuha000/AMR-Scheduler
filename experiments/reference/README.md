# 已发表口径的历史仿真参考源

amr-f62c064.zip 是项目提交 f62c0641bd0d56c86ddcf21d06e725c75d5b6a0f 的 amr_sched 目录原样归档。SHA-256：a9f249557b742f339331ccd781d8ed66d37fcc6be14c88d6d20020c158a0f164。归档命令：git archive --format=zip --output=amr-f62c064.zip f62c0641bd0d56c86ddcf21d06e725c75d5b6a0f amr_sched。

此归档用于复算既有结果表抽样和两条旧容量数值断言。辅助入口 tools/research_baseline.py 使用单独的 Python 进程，并核对归档摘要及加载路径。旧 CSV、指标断言、源版本和失败证据均保留；历史仿真不用于运行当前调度、ROS 2 或车辆控制。这里的抽样复算不代表历史全部 CSV 已重测，也不为旧版的共享通道容量问题提供通过结论。
完整历史研究源另存 amr-study-f62c064.zip，包含相同提交的 amr_sched、experiments、scenes。SHA-256：66888c0db6a7cf746f6b38df6d3a42d3caef46ed6f0843bf54bb3bbd09f8ca88。python tools/reproduce_historical.py --stage quick 在临时目录运行归档内原脚本，输出到新的 results/reproduce 子目录，并比较参考 CSV；原文件不移动或覆写。首次实跑 matrix.csv 和 baseline.csv 各 180 行全部一致，原 34 份 CSV 摘要保持。全量历史研究复算尚未执行，旧工具 tools/reproduce.py 仍使用当前工作树，不用于证明旧数值一致。
