"""pytest 共享配置：让测试无需安装即可 import fullstack_opd_v2（打包后可省略）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 高核服务器（96 核+）上 torch 默认 intra-op 线程池在嵌套多线程（scheduler 4 线程 +
# 测试主线程同时前向）下偶发死锁（实测在 torch._transformer_encoder_layer_fwd 卡死）。
# 测试用 toy 模型极小，1 线程足够且确定性更好——在 torch import 前锁死 OMP 并收线程数。
os.environ.setdefault("OMP_NUM_THREADS", "1")
# 耗时修复（IMP-1）：HF 测试必须 hermetic——`test_stage0_teachers_hf_missing_ref_raises`
# 曾真发 from_pretrained 到 hf-mirror.com，网络不可达时超时（7-50s，坏网络下可放大到 839s）。
# 所有 HF 相关测试都 mock from_pretrained / 用假路径期待失败，设离线后失败即返回，不触网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch  # noqa: E402
torch.set_num_threads(1)
