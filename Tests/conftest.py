"""Let pytest import amr_sched directly and avoid Windows WMI COM deadlocks on scientific imports."""

import os
import platform
import sys

if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
    platform._wmi = None  # type: ignore[attr-defined]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
