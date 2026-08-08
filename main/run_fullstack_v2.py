"""全栈 OPD 叠加 demo v2 入口（薄包装，转调 fullstack_opd_v2.demo.main）。

打包后（pip install -e .）无需本脚本，直接 `python -m fullstack_opd_v2`。
未安装时本脚本自动把源码目录加入 sys.path 作为回退。
"""
from __future__ import annotations

try:
    from fullstack_opd_v2.demo import main
except ImportError:                      # 未安装包时回退到源码目录
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fullstack_opd_v2.demo import main


if __name__ == "__main__":
    main()
