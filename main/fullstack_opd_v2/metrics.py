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
                 wandb_project: str | None = None, csv_path: str | None = None):
        self.backend = backend
        self.csv_path = csv_path or (os.path.join(run_dir, "metrics.csv")
                                     if run_dir else None)
        self._rows: list[dict] = []
        self._fields: list[str] = []
        self._file = None
        self._writer = None
        self._wandb = None
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
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=[])
        self._header_written = False
        self._fields = []

    def _ensure_fields(self, m: dict):
        for k in m:
            if k not in self._fields:
                self._fields.append(k)
        if self._writer is not None:
            self._writer.fieldnames = self._fields
            if not self._header_written:
                self._writer.writeheader()
                self._header_written = True

    # --------------------------- 记录 ---------------------------
    def record(self, m: dict):
        with self._lock:
            self._rows.append(dict(m))
            if self.backend == "csv" and self._writer is not None:
                self._ensure_fields(m)
                self._writer.writerow(m)
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