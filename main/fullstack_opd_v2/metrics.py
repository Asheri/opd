"""指标追踪：每步指标落 CSV + 可选 WandB，供训练后分析与跨 run 对比。

- `backend="csv"`（默认）：写 `run_dir/metrics.csv`，首条记录定列，后续补全/补齐。
- `backend="wandb"`：经 wandb 上传；wandb 未安装时**自动 fallback 到 CSV**（不崩）。
- `summary()`：汇总 n / 各数值字段均值，供训练结束打印健康信号。
"""

from __future__ import annotations

import csv
import os
import statistics
import threading
import warnings

from .exceptions import TrainingError


class MetricsRecorder:
    """每步指标记录器。"""

    def __init__(self, backend: str = "csv", run_dir: str | None = None,
                 wandb_project: str | None = None, csv_path: str | None = None,
                 flush_every: int = 10, append: bool = False):
        self.backend = backend
        self.csv_path = csv_path or (os.path.join(run_dir, "metrics.csv")
                                     if run_dir else None)
        self._rows: list[dict] = []
        self._fields: list[str] = []
        self._file = None
        self._writer = None
        self._wandb = None
        # L1：append=True 时对已存在的 metrics.csv 续写（resume 同 run_dir 保留历史），
        # 否则 "w" 截断（首跑/新 run）。
        self.append = bool(append)
        # C2：每 N 步才 flush 一次（默认 10），close 时终刷；避免每步一次系统调用
        self._flush_every = max(1, int(flush_every))
        self._n_records = 0
        # C1：后台消费线程与主线程可能并发 record/close，统一加锁保护
        self._lock = threading.Lock()

        if backend == "wandb":
            try:
                import wandb
                self._wandb = wandb.init(project=wandb_project or "opd")
            except Exception as e:
                warnings.warn(f"wandb 不可用，fallback 到 CSV：{e}")
                self.backend = "csv"
        elif backend not in ("csv", "none"):
            raise TrainingError(f"未知 metrics.backend={backend!r}（csv|wandb|none）")

        if self.backend == "csv" and self.csv_path:
            self._open_csv()

    # --------------------------- CSV ---------------------------
    def _open_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        exists = (os.path.isfile(self.csv_path)
                  and os.path.getsize(self.csv_path) > 0)
        mode = "a" if (self.append and exists) else "w"
        self._file = open(self.csv_path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=[])
        if self.append and exists:
            # 续写：读旧表头作为已有字段，不重复写表头（resume 保留历史）
            with open(self.csv_path, encoding="utf-8") as _f:
                self._fields = next(csv.reader(_f), [])
            self._writer.fieldnames = self._fields   # DictWriter 立即对齐旧列
            self._header_written = True
        else:
            self._header_written = False
            self._fields = []

    def _ensure_fields(self, m: dict) -> bool:
        """把 m 中出现的新字段并入 _fields 与表头。

        D2：表头按「首条记录定列」写出后，若后续记录才出现新字段（如 age），
        原地补列不可行（DictWriter 按 fieldnames 顺序写、旧行已落盘），改为
        全量重写（表头 + 已记录行）。返回 True 表示已整表重写（调用方须跳过
        本行的 writerow，避免重复写入）；False 表示按常规单行写入。
        """
        new = [k for k in m if k not in self._fields]
        if not new:
            return False
        # L1：append（resume 续写）模式下不整表重写（_rewrite_csv 会截断旧历史行）——
        # 新字段跳过 CSV 列（DictWriter 只写 fieldnames 列，多余键静默丢弃），仅警告。
        # 正常 resume 同流水线 schema 一致，此分支不应触发。
        if self.append and self._header_written:
            warnings.warn(
                f"append 模式出现新字段 {new}：跳过 CSV 列（resume 同流水线 schema 应一致）")
            return False
        self._fields.extend(new)
        if self._writer is None:
            return False
        if not self._header_written:
            self._writer.fieldnames = self._fields
            self._writer.writeheader()
            self._header_written = True
            return False            # 首个表头刚写出：本行仍由调用方 writerow
        self._rewrite_csv()         # 表头已存在 + 新字段晚到 → 整表重建（含 _rows 全部行）
        return True

    def _rewrite_csv(self):
        """按当前 _fields 重建整个 CSV：表头 + _rows 全部行，并落盘当前内容。"""
        self._file.seek(0)
        self._file.truncate()
        self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
        self._writer.writeheader()
        for row in self._rows:
            self._writer.writerow(row)
        self._file.flush()

    # --------------------------- 记录 ---------------------------
    def record(self, m: dict):
        with self._lock:
            self._rows.append(dict(m))
            if self.backend == "csv" and self._writer is not None:
                rewritten = self._ensure_fields(m)
                if not rewritten:                 # 常规路径：单行追加；整表重写则跳过
                    self._writer.writerow(m)
                # C2：每 N 步 flush 一次，避免每步一次系统调用；close 时终刷兜底
                self._n_records += 1
                if self._n_records % self._flush_every == 0:
                    self._file.flush()
            if self._wandb is not None:
                self._wandb.log(m)

    # --------------------------- 汇总 ---------------------------
    def summary(self) -> dict:
        with self._lock:
            s = {"n": len(self._rows)}
            if not self._rows:
                return s
            keys = self._rows[0].keys()
            for k in keys:
                vals = [r[k] for r in self._rows
                        if isinstance(r.get(k), (int, float)) and not isinstance(r[k], bool)]
                if vals:
                    s[f"{k}_mean"] = statistics.mean(vals)
                    s[f"{k}_last"] = vals[-1]
            return s

    def close(self):
        with self._lock:
            if self._file is not None:
                # C2：终刷兜底——最后一次不满 flush_every 的记录也落盘。
                # 注意不显式调 flush：file.close() 内部自带一次 flush（CPython
                # close 会 flush 缓冲区），显式再调会让 flush spy 多计一次无关的
                # 隐式调用（C2 测试按「节流 1 次 + close 终刷 1 次」= 2 断言）。
                self._file.close()
                self._file = None
            if self._wandb is not None:
                try:
                    self._wandb.finish()
                except Exception:
                    pass
                self._wandb = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["MetricsRecorder"]