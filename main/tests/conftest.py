"""pytest 共享配置：让测试无需安装即可 import fullstack_opd_v2（打包后可省略）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
