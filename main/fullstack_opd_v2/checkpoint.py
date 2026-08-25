"""checkpoint 管理：学生模型断点保存/加载/续跑。

工程化核心：训练可中断、可续跑。每 `every` 步把
`state_dict + version + step + cfg（+ 最近 metrics）` 落盘到
`run_dir/checkpoints/step_<N>.pt`，`--resume` 时加载最新断点续跑。
"""

from __future__ import annotations

import os
import re

import torch

from .exceptions import CheckpointError


def _release_cpu_memory() -> None:
    """checkpoint 保存后归还 CPU 内存（E1 SIGKILL 根因缓解）。

    torch.save 的 CPU payload 可达 13.5GB，PyTorch CPU allocator 缓存 + glibc malloc
    arena 碎片使单进程 RSS 峰值 206GB（占 cgroup 220GB 配额 94%）。gc.collect +
    malloc_trim 强制归还空闲页；非 glibc / 非 Linux 平台静默跳过，绝不抛异常。
    """
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _opt_state_to_cpu(optimizer) -> dict:
    """把 optimizer state 中的张量搬 CPU（§B 精确续跑，供 checkpoint 落盘）。

    单卡直接读 state_dict；FSDP/分布式下应改用 `optimizer.get_state_dict()`（对称恢复
    用 `optimizer.load_state_dict` / `set_state_dict`）。返回的结构与 torch.optim
    state_dict 一致，仅所有 value 张量 .cpu()。
    """
    state = {}
    opt_sd = getattr(optimizer, "get_state_dict", None)
    if callable(opt_sd):
        sd = opt_sd()
    else:
        sd = optimizer.state_dict()
    state["state"] = {
        k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
            for kk, vv in v.items()}
        for k, v in sd["state"].items()}
    state["param_groups"] = sd.get("param_groups", [])
    return state


class CheckpointManager:
    """管理 run_dir/checkpoints/ 下的断点，支持节流保存与最新定位。"""

    def __init__(self, run_dir: str, every: int = 10,
                 checkpoint_dir: str | None = None):
        self.checkpoint_dir = checkpoint_dir or os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.every = max(1, int(every))

    # --------------------------- 保存 ---------------------------
    def save(self, step: int, student, version: int, cfg: dict,
             metrics: list | None = None, force: bool = False,
             ref: dict | None = None,
             optimizer=None, rng: dict | None = None,
             refresh_buffer=None) -> str | None:
        """若 step 是 every 的倍数则存断点，否则跳过（节流）。force=True 无条件存。

        ref: Stage 2 的 KL 锚点打包字典 `{"ref_dists"/"ref_ids"/"ref_logp"}`（初始 student
             在 fat D 上的分布）。随断点落盘，resume 时直接恢复，避免用已训练 student
             重算锚点破坏「KL 锚点 = 初始 student 分布」不变式（A3/D4）。

        §B 精确续跑（L2，任务 6.1）：optimizer（optimizer state → CPU）、rng（RNG 状态）、
        refresh_buffer（L2 ring buffer）。三者皆为可选；None 时断点不含相应键（旧断点兼容）。
        """
        if step % self.every != 0 and not force:
            return None
        path = os.path.join(self.checkpoint_dir, f"step_{step}.pt")
        tmp = path + ".tmp"
        payload = {
            "step": step,
            "version": version,
            "state": {k: v.detach().cpu() for k, v in student.state_dict().items()},
            "cfg": cfg,
            "metrics": (metrics or [])[-1:],
            "ref": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in (ref or {}).items()},
        }
        if optimizer is not None:
            payload["optimizer"] = _opt_state_to_cpu(optimizer)
        if rng is not None:
            payload["rng"] = rng
        if refresh_buffer is not None:
            payload["refresh_buffer"] = refresh_buffer
        torch.save(payload, tmp)
        os.replace(tmp, path)                     # 原子替换，避免半写
        # cgroup 内存修复（2026-08-25）：checkpoint 的 CPU payload 可达 13.5GB，
        # torch.save 后不归还 → PyTorch CPU allocator 缓存 + glibc malloc arena 碎片
        # 使单进程 RSS 峰值 206GB（占 cgroup 220GB 配额 94%），双进程并行必被 SIGKILL。
        # 保存后立即释放 payload 并强制归还空闲页（E1 SIGKILL 根因的缓解）。
        del payload
        _release_cpu_memory()   # gc.collect + malloc_trim（cgroup 内存修复，见函数 docstring）
        return path

    # --------------------------- 定位 ---------------------------
    def latest(self) -> str | None:
        """返回最新（step 最大）的断点路径。"""
        best = None
        best_step = -1
        if os.path.isdir(self.checkpoint_dir):
            for name in os.listdir(self.checkpoint_dir):
                m = re.match(r"step_(\d+)\.pt$", name)
                if m and int(m.group(1)) > best_step:
                    best_step = int(m.group(1))
                    best = os.path.join(self.checkpoint_dir, name)
        return best

    # --------------------------- 加载 ---------------------------
    def load(self, ckpt_path: str) -> dict:
        """加载断点 → {step, version, state, cfg, metrics, ref}。ref 为 KL 锚点（旧断点无）。"""
        if not os.path.isfile(ckpt_path):
            raise CheckpointError(f"断点不存在: {ckpt_path}")
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise CheckpointError(f"加载断点失败 {ckpt_path}: {e}") from e
        return ck

    def resume(self) -> dict | None:
        """自动定位最新断点并加载；无断点返回 None。"""
        latest = self.latest()
        if latest is None:
            return None
        return self.load(latest)


__all__ = ["CheckpointManager"]
