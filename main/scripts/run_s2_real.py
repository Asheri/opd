#!/usr/bin/env python3
"""S2 真实 GPU 实验：加载 skywork_17b.yaml 基座 + STAGE2_ROLLOUT_MATRIX 覆盖。

跑真实模型 S2_E0/E1/E2/E3（真实 512/1024/2048 rollout），产出训练 summary
（reward/pg_loss/kl_loss + rollout/* 状态计数），供 report_stage2 生成 Q1-Q4。

用法（串行，默认）：
  python run_s2_real.py --config configs/skywork_17b.yaml --run-dir <dir> \
      --device cuda:0 --n-steps 30 \
      [--names S2_E0_static S2_E1_opd512 S2_E2_opd1024 S2_E3_opd2048] \
      [--eos-id 151645] [--materialized 500]

用法（--parallel N>1，双卡交叉分卡并行，AGENTS.md GPU≥2 硬约束）：
  python run_s2_real.py --config configs/skywork_17b.yaml --run-dir <dir> \
      --names S2_E1_opd512 S2_E2_opd1024 --parallel 2 \
      --batch-size 2 --set stage2.offload_to_cpu=true \
      --set stage2.queue_size=2 --set stage2.staleness_queue_min=2

  - 第 i 个实验子进程训练卡 cuda:{i%N}、vLLM 引擎 rollout_device cuda:{(i+1)%N}
    （交叉分卡：NCCL 硬约束 trainer 与 vLLM worker 必须不同物理卡）。
    E1（train cuda:0）不用写 rollout_device（默认 cuda:1）；E2（train cuda:1）
    由 --parallel 自动反写 --set stage2.rollout_device=cuda:0。
  - 子进程全部强制 --load-cache 只读复用预建缓存；缓存缺失时父进程先串行构建一次
    （首实验，仅一次），随后所有子进程并发读 mmap。
  - 子进程写 run-dir/<实验名>/summary.json，父进程 join 后合并成
    run-dir/l2_experiment_summary.json。
  - 只能 multiprocessing spawn（fork 与 vLLM/NCCL 不兼容，vLLM 强制 spawn）。

与 toy 端 run_matrix 的关键差异：
  - 基座是真实 YAML（model_kind=hf + 真实模型/数据/教师对），非 DEFAULT_CONFIG_V2。
  - 每个实验独立 run 目录；共享同一份预建教师缓存（首实验 load_cache=false 建，
    其后实验 load_cache=true 复用，避免重复 GPU 建缓存）。
  - 单实验 try/except 隔离：一个失败不中断矩阵。
  - 产出每实验 {name, summary, run_dir} 并入 l2_experiment_summary.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# 脚本位于 main/scripts/ 下；直接运行（python scripts/xxx.py）时 sys.path[0] 是 scripts/
# 而非 repo 根，导致 `from fullstack_opd_v2 ...` 失败。显式把 main/ 加入 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.experiment import STAGE2_ROLLOUT_MATRIX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="基座 YAML（skywork_17b.yaml）")
    p.add_argument("--run-dir", required=True, help="输出根目录")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--names", nargs="+", default=None,
                   help="实验名（默认全矩阵）")
    p.add_argument("--eos-id", type=int, default=None,
                   help="l2.rollout.eos_token_id（校准值 151645）")
    p.add_argument("--materialized", type=int, default=0,
                   help="base.materialized_size（预生成 response 的锚点数）")
    p.add_argument("--m-refresh", type=int, default=8,
                   help="l2.m_refresh（每刷新相位的 rollout 条数）")
    p.add_argument("--refresh-min", type=int, default=10,
                   help="l2.cache.refresh_min_interval（步数间隔触发刷新）")
    p.add_argument("--cache-path", default=None,
                   help="教师缓存路径（覆盖 YAML）")
    p.add_argument("--load-cache", action="store_true",
                   help="复用已建缓存（首实验建后置 true）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="覆盖 stage2.batch_size（真实 (4,3072,151936) 序列 flash 后仍 ~87GB，"
                        "batch 4→2 把训练激活减半防 OOM；默认继承 config）")
    p.add_argument("--refresh-size", type=int, default=None,
                   help="覆盖 l2.cache.refresh_size（默认 5000×T×K 预分配 GPU OOM，pilot 用 ~64）")
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="并行实验数：N>1 时每个实验 spawn 一个子进程，按 N 张卡交叉分卡"
                        "（第 i 个实验训练 cuda:{i%N}、vLLM rollout_device cuda:{(i+1)%N}），"
                        "全部子进程强制 --load-cache 只读复用预建缓存；默认 1=串行（不改原行为）")
    p.add_argument("--stagger", type=float, default=0.0, metavar="SEC",
                   help="--parallel N>1 时，子进程间启动间隔秒数（默认 0=同时启动）。"
                        "vLLM init 与对方训练峰值竞态时设 30-60 可确定性消除"
                        "（v6 实测：E1 的 vLLM 在 GPU1 init 时 E2 已占 71GB -> profiling 失败）")
    # 内部标记：仅由父进程 --parallel 路径追加到子进程 argv，用户无需使用
    p.add_argument("--parallel-child", action="store_true", help=argparse.SUPPRESS,
                   dest="parallel_child")
    p.add_argument("--set", dest="extra_sets", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="额外 config 覆盖（可重复），如 --set stage2.rollout_engine=vllm")
    p.add_argument("--resume", action="store_true", help="从 run-dir 最新断点续跑剩余步（需与断点同配置；metrics 截断到断点前，step 编号不重复）")
    return p.parse_args()


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def _printer(prefix):
    """返回带 [S2:<prefix>] 前缀的打印函数；prefix 为 None 时与原生 print 等价（串行零变化）。

    带前缀（并行子进程 / 父进程 cache 构建）时做窄编码控制台兜底：GBK 等编码打不出
    emoji（如 ❌）会抛 UnicodeEncodeError，用 errors="replace" 降级保证输出与
    summary.json 不被吞掉真实异常文本。
    """
    if not prefix:
        return print

    def _pfx(*args, **kwargs):
        text = f"[S2:{prefix}] " + " ".join(str(a) for a in args)
        try:
            print(text, **kwargs)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(text.encode(enc, "replace").decode(enc, "replace"), **kwargs)

    return _pfx


# ---------------------------------------------------------------- 单实验执行 ---
def _build_overrides(args, name, load_cache):
    """单实验 config 覆盖列表（与改造前逐字一致：矩阵 + 通用长度/缓存开关 + --set 透传）。"""
    if name not in STAGE2_ROLLOUT_MATRIX:
        raise SystemExit(f"未知实验 {name!r}，可选 {list(STAGE2_ROLLOUT_MATRIX)}")
    overrides = [f"{k}={v}" for k, v in STAGE2_ROLLOUT_MATRIX[name].items()]
    overrides += [
        f"stage2.n_steps={args.n_steps}",
        f"l2.m_refresh={args.m_refresh}",
        f"l2.cache.refresh_min_interval={args.refresh_min}",
        f"l2.cache.refresh_max_interval={args.refresh_min + args.n_steps}",
    ]
    if args.batch_size:
        overrides.append(f"stage2.batch_size={args.batch_size}")
    if args.refresh_size:
        overrides.append(f"l2.cache.refresh_size={args.refresh_size}")
    if args.eos_id is not None:
        overrides.append(f"l2.rollout.eos_token_id={args.eos_id}")
    if args.materialized:
        overrides.append(f"base.materialized_size={args.materialized}")
    if args.cache_path:
        overrides.append(f"stage1.cache_path={args.cache_path}")
    overrides.append(f"stage1.load_cache={'true' if load_cache else 'false'}")
    overrides += list(args.extra_sets)   # --set 透传：rollout_engine=vllm 等任意覆盖
    return overrides


def _truncate_metrics_csv(run_dir: str, resume_step: int, backup: bool = True) -> None:
    """resume 续跑前，把旧 metrics.csv 截断到断点 step 之前（避免续跑 step 编号重复）。

    只保留 step < resume_step 的行（空 step 行也保留）；无 metrics 文件时静默跳过。
    backup=True（默认）：截断前先把完整 metrics.csv 备份为
    `metrics_pre_resume_step<N>.csv`（已存在则不覆盖）——训练产物不可再生约束
    （AGENTS.md）：metrics 一旦丢失无法重建，resume 重复截断 + 清理失误曾让 E1
    丢失 120 步 eval_reward（2026-08-26 教训），备份保证历史曲线始终可恢复。
    """
    import csv as _csv
    import shutil
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(p):
        return
    if backup:
        bak = os.path.join(run_dir, f"metrics_pre_resume_step{resume_step}.csv")
        if not os.path.isfile(bak):
            shutil.copyfile(p, bak)
    with open(p, encoding="utf-8", newline="") as f:
        reader = list(_csv.DictReader(f))
    if not reader:
        return
    fieldnames = list(reader[0].keys())
    keep = [r for r in reader
            if not r.get("step") or not r["step"].strip()
            or int(r["step"]) < resume_step]
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(keep)


def _run_experiment(args, name, load_cache, prefix=None):
    """串行跑单个实验；返回 results 元素 {name, summary, run_dir}。

    与改造前语义完全一致：config 错误（load_config）直接抛出；训练异常 try/except
    隔离成 error summary 不中断矩阵。
    """
    cfg = load_config(path=args.config, overrides=_build_overrides(args, name, load_cache))

    d = os.path.join(args.run_dir, name)
    os.makedirs(d, exist_ok=True)
    log = _printer(prefix)
    log(f"\n===== 实验 {name} =====  (n_steps={args.n_steps}, "
        f"materialized={args.materialized}, load_cache={load_cache})", flush=True)
    try:
        from fullstack_opd_v2.pipeline import FullStackOPDv2
        resume = None
        if args.resume:
            from fullstack_opd_v2.checkpoint import CheckpointManager
            cm = CheckpointManager(d, every=int((cfg.get("run") or {}).get("checkpoint_every", 10)))
            resume = cm.resume()
            if resume:
                _rs = int(resume.get("step", resume.get("version", 0)))
                _res_v = resume.get("version", 0)
                log(f"  [resume] 从断点 step={_rs} 续跑（version={_res_v}）", flush=True)
                # 训练产物不可再生约束（AGENTS.md）：续跑前截断旧 metrics 到断点前
                # （避免重复 step 行）；截断前自动备份 metrics_pre_resume_step<N>.csv。
                _truncate_metrics_csv(d, _rs)
            else:
                log("  [resume][警告] run-dir 无断点，从 0 开始", flush=True)
        out = FullStackOPDv2(cfg, device=args.device).run(run_dir=d, resume=resume)
        metrics = out["metrics"]   # 8d1411c 重写时丢行（b4b9872 漏修）：缺此行汇总 NameError
        # M3：均值只统计【含该键】的训练步 metric——rollout 相位 metric 缺键时
        # 旧实现 m.get(k, 0.0) 会往 reward/pg/kl 均值里混入大量 0，污染口径。
        def _keyed_mean(key):
            vals = [m[key] for m in metrics
                    if isinstance(m, dict) and key in m]
            return _mean(vals)
        summary = {
            "experiment": name,
            "n_steps": sum(1 for m in metrics
                           if isinstance(m, dict) and m.get("phase") != "rollout"),
            "reward_mean": _keyed_mean("reward"),
            "pg_loss_mean": _keyed_mean("pg_loss"),
            "kl_loss_mean": _keyed_mean("kl_loss"),
            "total_s": round(out["timings"].get("total", 0.0), 3),
        }
        # rollout 状态计数（最后一个 refresh 相位）
        for col, key in [("rollout/n_appended", "rollout_n_appended"),
                         ("rollout/n_eos", "rollout_n_eos"),
                         ("rollout/n_loop", "rollout_n_loop")]:
            for m in reversed(metrics):
                if isinstance(m, dict) and col in m:
                    summary.setdefault(key, m[col])
                    break
        log(f"  summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
        return {"name": name, "summary": summary, "run_dir": d}
    except Exception as e:
        import traceback
        log(f"  ❌ 实验 {name} 失败: {e}", flush=True)
        traceback.print_exc()
        return {"name": name, "summary": {"experiment": name, "error": str(e)},
                "run_dir": d}


# ---------------------------------------------------------------- 并行工具（纯函数，供单测）---
_STRIP_OPTIONS = ("--names", "--device", "--parallel", "--stagger")


def cgroup_memory_warning(n_parallel: int, per_proc_gb: float = 210.0,
                        cgroup_path: str = "/sys/fs/cgroup/memory.max") -> str | None:
    """读 cgroup 内存配额（默认 /sys/fs/cgroup/memory.max），若 并行数×单进程峰值 > 配额返回警告。
    三态：无 cgroup 文件 / 配额=max → None；配额内 → None；超限 → 警告文案。
    E1 SIGKILL 根因（2026-08-25）：容器 cgroup 内存硬限 220GB，单进程 RSS 峰值 206GB，
    双进程并行 checkpoint 保存时超限 → cgroup OOM killer 发 SIGKILL（exitcode=-9）。
    """
    try:
        with open(cgroup_path, encoding="utf-8") as f:
            raw = f.read().strip()
        if raw in ("max", ""):
            return None
        quota = int(raw)
    except (OSError, ValueError):
        return None
    peak = n_parallel * per_proc_gb * (1024 ** 3)
    if peak > quota:
        gb = quota / (1024 ** 3)
        return ("[S2][警告] cgroup 内存配额 {:.0f}GB，{} 进程 × 预估峰值 {:.0f}GB = {:.0f}GB 超限"
                " —— 建议串行跑（或降低 batch/offload）。E1 曾因双并行 + checkpoint CPU payload"
                " 超 220GB 被 SIGKILL（exitcode=-9）。").format(gb, n_parallel, per_proc_gb, peak / (1024 ** 3))
    return None
def build_parallel_argv(base_argv, name, i, n_cards):
    """父进程 sys.argv → 单个并行子进程 argv（纯函数，单测覆盖）。

    - 剔除 --names/--device/--parallel（含 `--opt value` 与 `--opt=value` 两种写法；
      --names 的 nargs='+' 值列表被整体吸收），其余参数原样保留、顺序不变
      → --batch-size / --set / --cache-path / --eos-id 等显存收敛与部署配置自动透传；
    - 追加 --names <name>（该实验黑名）、--device cuda:{i%n}（第 i 张卡）、
      --set stage2.rollout_device=cuda:{(i+1)%n}（交叉分卡：trainer 与 vLLM
      必须不同物理卡）、--load-cache（强制只读复用预建缓存）。
    """
    if n_cards < 1:
        raise ValueError("n_cards 必须 >= 1")
    out = []
    j, n = 0, len(base_argv)
    while j < n:
        tok = base_argv[j]
        opt = tok.split("=", 1)[0]
        if opt in _STRIP_OPTIONS:
            # --names 是 nargs="+"：值列表整体吸收到下一个 "-" 开头的 token
            # （"--names=A B" 混合写法同样吸收，等号值已在 token 内）
            if opt == "--names":
                j += 1
                while j < n and not base_argv[j].startswith("-"):
                    j += 1
            elif "=" in tok:      # --device=cuda:0 / --parallel=2 单值等号形式
                j += 1
            else:                 # --device cuda:0 / --parallel 2
                j = min(j + 2, n)
            continue
        out.append(tok)
        j += 1
    out += ["--names", name,
            "--device", f"cuda:{i % n_cards}",
            "--set", f"stage2.rollout_device=cuda:{(i + 1) % n_cards}",
            "--load-cache"]
    return out


def merge_summaries(run_dir):
    """扫描 run_dir 下各实验子目录的 summary.json，合并为与 results 同构的列表。

    每个元素 {name, summary, run_dir}（与串行路径 results 结构一致）；缺失/损坏的
    summary.json（非对象 / 无 summary 字段 / json 解析失败）跳过并在 stdout 警告。
    按目录名排序保证合并结果确定性。
    """
    entries = []
    if not os.path.isdir(run_dir):
        return entries
    for name in sorted(os.listdir(run_dir)):
        d = os.path.join(run_dir, name)
        if not os.path.isdir(d):
            continue
        summary_path = os.path.join(d, "summary.json")
        if not os.path.isfile(summary_path):
            print(f"  [S2][警告] {name}/summary.json 缺失，跳过", flush=True)
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("非 JSON 对象")
            summary = data.get("summary")
            if not isinstance(summary, dict):
                raise ValueError("缺 summary 字段")
            entries.append({
                "name": data.get("name") or data.get("experiment") or name,
                "summary": summary,
                "run_dir": d,
            })
        except Exception as e:
            print(f"  [S2][警告] {name}/summary.json 损坏（{e}），跳过", flush=True)
    return entries


# ---------------------------------------------------------------- 汇总落盘 ---
def _write_shared_summary(args, results):
    """串行路径：与改造前一致的共享汇总（dict：{实验名: summary}）。"""
    out_path = os.path.join(args.run_dir, "l2_experiment_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({r["name"]: r["summary"] for r in results}, f, indent=2,
                  ensure_ascii=False)
    print(f"\n✅ 汇总写入 {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def _write_experiment_summary(result):
    """并行路径：把单实验结果写为 run-dir/<实验名>/summary.json（父进程合并用）。"""
    path = os.path.join(result["run_dir"], "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- 并行执行 ---
def _cache_exists(cache_path):
    """预建缓存是否存在：memory 单 .pt 文件 / disk <prefix>.metadata.json 前缀（任一存在即可）。"""
    return bool(cache_path) and (
        os.path.isfile(cache_path) or os.path.isfile(f"{cache_path}.metadata.json"))


def _resolve_cache_path(args, name):
    """第一实验实际使用的教师缓存路径（--cache-path 优先，否则按同序覆盖从配置读出）。"""
    if args.cache_path:
        return args.cache_path
    overrides = _build_overrides(args, name, load_cache=False)
    cfg = load_config(path=args.config, overrides=overrides)
    return cfg["stage1"].get("cache_path")


def _spawn_entry(child_argv):
    """并行子进程入口：spawn 出的全新解释器里用子 argv 重建 sys.argv 后跑 main。

    spawn 子进程以 __mp_main__ 重新执行本脚本（__main__ 保护下 main 不会自动跑），
    再由 multiprocessing 反序列化调用本入口。
    """
    sys.argv = list(child_argv)
    try:
        main()
    finally:
        # 双引擎并发时 vLLM 引擎 shutdown/解释器退出清理会挂起（非 daemon 线程
        # join），child 永不退出 → 父 join 永久阻塞、EngineCore 变孤儿占显存
        # （2026-08-19 双卡实测）。summary.json 在并行子路径也已写完，此处强制
        # 退出跳过解释器清理；残留 EngineCore 由父进程 join 后统一清理。
        os._exit(0)


def _run_serial(args, names):
    """N=1 串行路径：逐个跑实验（try/except 隔离），保持改造前语义与输出。"""
    results = []
    for name in names:
        prefix = name if args.parallel_child else None
        try:
            results.append(_run_experiment(args, name, load_cache=args.load_cache,
                                           prefix=prefix))
        except Exception as e:
            if not args.parallel_child:
                raise   # N=1 串行：load_config 等配置错误保持原样向上传播（改造前语义）
            # 并行子进程兜底：单实验内部除处理器再抛（如 GBK 控制台打不出 ❌ emoji）
            # 时，仍写成含 error 的 summary.json 并以 0 退出，避免整表丢失。
            import traceback
            log = _printer(prefix)
            log(f"  [S2][警告] 实验 {name} 异常未捕获: {e}", flush=True)
            traceback.print_exc()
            d = os.path.join(args.run_dir, name)
            os.makedirs(d, exist_ok=True)
            results.append({"name": name,
                            "summary": {"experiment": name, "error": str(e)},
                            "run_dir": d})
    if args.parallel_child:
        # 并行子进程：只写本实验 summary.json，绝不写共享 l2_experiment_summary.json
        for r in results:
            _write_experiment_summary(r)
        return
    _write_shared_summary(args, results)


def _finish_parallel(args, n_failed=0):
    """join 全部子进程后：合并各实验 summary.json 为共享汇总（list[{name,summary,run_dir}]）。"""
    results = merge_summaries(args.run_dir)
    out_path = os.path.join(args.run_dir, "l2_experiment_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[S2] 汇总写入 {out_path}（{len(results)} 个实验，{n_failed} 个子进程非零退出）", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def _run_parallel(args, names):
    """--parallel N>1：为每个实验 spawn 一个子进程，交叉分卡并行（GPU≥2 硬约束）。

    - 第 i 个实验子进程 --device cuda:{i%N}，追加 --set stage2.rollout_device=cuda:{(i+1)%N}
      （NCCL 硬约束：trainer 与 vLLM worker 必须不同物理卡）；
    - 全部子进程强制 --load-cache；预建缓存缺失时父进程先串行构建一次（首实验，
      load_cache=false，仅一次），子进程只读 mmap 复用（并发安全）；
    - 子进程写 run-dir/<实验名>/summary.json；父进程 join 后 merge_summaries 合并。
    """
    import multiprocessing   # 仅在并行路径导入：N=1 串行路径无该依赖（避免不必要副作用）
    for name in names:
        if name not in STAGE2_ROLLOUT_MATRIX:
            raise SystemExit(f"未知实验 {name!r}，可选 {list(STAGE2_ROLLOUT_MATRIX)}")

    cache_path = _resolve_cache_path(args, names[0])
    names_to_spawn = list(names)
    if not _cache_exists(cache_path):
        print(f"\n[S2] 预建缓存 {cache_path!r} 不存在——父进程先串行构建一次"
              f"（首实验 {names[0]}，其余子进程 --load-cache 只读复用）", flush=True)
        result = _run_experiment(args, names[0], load_cache=False, prefix=names[0])
        _write_experiment_summary(result)
        names_to_spawn = names[1:]
        if not names_to_spawn:
            _finish_parallel(args)
            return

    ctx = multiprocessing.get_context("spawn")   # vLLM/NCCL 强制 spawn（fork 不兼容）
    procs = []
    for i, name in enumerate(names_to_spawn):
        # vLLM init 错峰（--stagger）：第 2+ 个子进程延迟启动，避免其 vLLM 引擎在
        # 对方训练峰值时 profiling 失败（v6 实测：E1 的 vLLM 在 GPU1 init 时 E2 已占
        # 71GB -> "No available memory for cache blocks"）。
        if i > 0 and args.stagger > 0:
            print(f"[S2] stagger: 等待 {args.stagger:.0f}s 后启动第 {i + 1} 个子进程"
                  f"（{name}，vLLM init 错峰）...", flush=True)
            time.sleep(args.stagger)
        child_argv = build_parallel_argv(sys.argv, name, i, args.parallel)
        child_argv += ["--parallel-child"]
        p = ctx.Process(target=_spawn_entry, args=(child_argv,), name=f"S2-{name}")
        p.start()
        procs.append((name, p))

    n_failed = 0
    for name, p in procs:
        p.join()
        # 2026-08-22：子进程 os._exit 跳过清理 → 其 EngineCore 变孤儿占显存。立即清理
        # 孤儿（ppid=1），否则先完成的实验引擎一直占着目标卡，拖累仍在跑的另一实验
        # （v13 实测 E1 残留 12.76GB 致 E2 refresh OOM）。
        _cleanup_orphan_engines()
        # P1（2026-08-22）：清理后检查是否仍有 EngineCore 孤儿（ppid=1）残留——
        # 若有，说明清理静默失效（如 ps 格式变化），打警告提示人工核查，不自动
        # pkill 全部（会误杀仍在跑实验的引擎）。
        _left = _count_orphan_engines()
        if _left > 0:
            print(f"  [S2][警告] 孤儿引擎清理后仍检测到 {_left} 个 ppid=1 的 "
                  "EngineCore，请检查 ps args= 输出格式", flush=True)
        if p.exitcode not in (0, None):
            n_failed += 1
            print(f"  [S2][警告] 子进程 {name} 非零退出（exitcode={p.exitcode}），计入失败", flush=True)
    _cleanup_stray_engines()
    _finish_parallel(args, n_failed=n_failed)



def _orphan_engine_pids() -> list[int]:
    """列出 ppid=1 且 cmdline 含 'VLLM::EngineCore' 的 pid（ps args= 读 cmdline 不截断）。

    v14 实测：comm= 截断 16 字符名为 15（VLLM::EngineCor），'VLLM::EngineCore' 子串
    恒不匹配 -> 清理从未执行；args= 读 /proc/PID/cmdline（setproctitle 修改后不截断）。
    纯函数，CPU 可单测（monkeypatch ps 输出）。
    """
    import subprocess as _sp
    out = _sp.run(["ps", "-eo", "pid=,ppid=,args="],
                  capture_output=True, text=True).stdout
    pids: list[int] = []
    for line in out.splitlines():
        if "VLLM::EngineCore" not in line:
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "1":   # ppid=1 → 子进程已退出
            pids.append(int(parts[0]))
    return pids


def _count_orphan_engines() -> int:
    """统计 ppid=1 的 EngineCore 孤儿数（不 kill）。供清理后检查防静默失效。"""
    try:
        return len(_orphan_engine_pids())
    except Exception:   # pragma: no cover —— 统计失败返回 -1 表示无法判定
        return -1


def _cleanup_orphan_engines() -> None:
    """join 后清理【孤儿】vLLM EngineCore（ppid=1，其子进程已 os._exit 退出）。

    与 _cleanup_stray_engines（pkill 全部）不同：只杀 ppid=1 的 EngineCore，不误杀仍
    在运行实验的引擎。并行双实验先完成的那个 os._exit 跳过清理 → 其 EngineCore 变孤儿
    继续占目标卡显存（v13 实测：E1 残留 12.76GB 致 E2 refresh OOM 80.94GB 撞顶）。
    每 join 一个子进程后立即调用，及时释放其引擎占用的显存，避免拖到全 join 后。
    """
    import os as _os
    try:
        for pid in _orphan_engine_pids():
            _os.kill(pid, 9)
    except Exception as e:   # pragma: no cover —— 清理失败只告警
        print(f"  [S2][警告] 孤儿引擎清理失败: {e}", flush=True)


def _cleanup_stray_engines() -> None:
    """join 后清理子进程留下的孤儿 vLLM EngineCore（child os._exit 跳过清理）。

    双卡并行：child 完成即 os._exit(0)（防 vLLM shutdown 挂起），引擎进程残留为
    init 孤儿（ppid=1）继续占显存；此处按 cmdline 精确匹配 kill。专用云 GPU 机器，
    pkill 'VLLM::EngineCore' 不匹配父/其他链路。
    """
    import subprocess
    try:
        subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:   # pragma: no cover —— 清理失败只告警，不影响汇总
        print(f"  [S2][警告] 残留引擎清理失败: {e}", flush=True)

def main() -> None:
    args = parse_args()
    names = args.names or list(STAGE2_ROLLOUT_MATRIX)
    os.makedirs(args.run_dir, exist_ok=True)
    if args.parallel > 1:
        # cgroup 内存配额断言（2026-08-25 E1 SIGKILL 根因）：双并行 × 206GB 峰值 > 220GB 配额
        _cw = cgroup_memory_warning(args.parallel)
        if _cw:
            print(_cw, flush=True)
        _run_parallel(args, names)
        return
        _run_parallel(args, names)
        return
    _run_serial(args, names)


if __name__ == "__main__":
    main()
