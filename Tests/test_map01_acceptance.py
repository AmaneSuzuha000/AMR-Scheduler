"""MAP-01 场景验收脚本的守护测试。

七项验收（连通性、任务合法性、物理通道共享、装卸后释放、几何可通行、
坐标正反变换、版本可追溯）是任务书明文列的条款。脚本自己会打印逐项结果，
这里只保证它整体是绿的——任何一项退化，这条测试就红。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import verify_map01


def test_map01_acceptance_all_seven_pass(capsys):
    rc = verify_map01.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "MAP-01 验收通过 7 / 7" in out, out
