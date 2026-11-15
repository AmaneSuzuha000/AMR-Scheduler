"""Let pytest import amr_sched directly (without relying on installation)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
