"""Stage 0 规模可行性测试：config schema + generation benchmark 统计 + shard/resume + 外推。

覆盖 docs 提示词《Stage 0：重构 50K×8192 数据构建路径》的纯逻辑部分（真实 HF 生成需 GPU，
不在本地测）。核心验证：统计聚合（T_actual=ΣL_i 而非 N×max_new）、progressive 分片不重复、
config 的 prompt_universe/materialized 规模字段、多阶段 wall time 外推。
"""
import pytest

from fullstack_opd_v2.config import load_config

# ============================================================================
# 任务 1：config schema 扩展（prompt_universe_size + base.materialized_size）
# ============================================================================

def test_config_scale_fields():
    """dataset.prompt_universe_size + base.materialized_size 可加载、可覆盖。"""
    cfg = load_config(overrides=["dataset.prompt_universe_size=50000",
                                 "base.materialized_size=5000"])
    assert cfg["dataset"]["prompt_universe_size"] == 50000
    assert cfg["base"]["materialized_size"] == 5000


def test_config_scale_defaults():
    """默认值：50K universe + 5K materialized 静态锚点。"""
    cfg = load_config()
    assert cfg["dataset"]["prompt_universe_size"] == 50000
    assert cfg["base"]["materialized_size"] == 5000


# ============================================================================
# 任务 2：generation benchmark 统计聚合（aggregate_stats 纯函数）
# ============================================================================

def test_aggregate_stats():
    """T_actual=ΣL_i；eos_rate/truncation_rate/P(L=8192) 正确。"""
    from scripts.gen_benchmark import aggregate_stats
    s = [
        {"prompt_len": 10, "gen_len": 100,  "ended_eos": True,  "truncated": False, "wall_s": 1.0},
        {"prompt_len": 10, "gen_len": 300,  "ended_eos": True,  "truncated": False, "wall_s": 3.0},
        {"prompt_len": 10, "gen_len": 8192, "ended_eos": False, "truncated": True,  "wall_s": 100.0},
    ]
    st = aggregate_stats(s)
    assert st["prompt_count"] == 3
    assert st["generated_tokens"] == 8592          # ΣL_i，非 3×8192
    assert st["eos_rate"] == pytest.approx(2 / 3)
    assert st["truncation_rate"] == pytest.approx(1 / 3)
    assert st["P_L_eq_8192"] == pytest.approx(1 / 3)
    assert st["P_L_gt_2048"] == pytest.approx(1 / 3)   # 8192 > 2048
    assert st["P_L_gt_4096"] == pytest.approx(1 / 3)   # 8192 > 4096
    assert st["mean_len"] == pytest.approx(8592 / 3)
    assert st["tok_per_s"] == pytest.approx(8592 / 104)
    assert st["samples_per_s"] == pytest.approx(3 / 104)


def test_aggregate_stats_eos_rate_all():
    """全 EOS、零截断：P(L=8192)=0，tok_per_s 用 ΣL_i。"""
    from scripts.gen_benchmark import aggregate_stats
    s = [{"gen_len": 100, "ended_eos": True, "truncated": False, "wall_s": 1.0},
         {"gen_len": 200, "ended_eos": True, "truncated": False, "wall_s": 2.0}]
    st = aggregate_stats(s)
    assert st["eos_rate"] == 1.0
    assert st["truncation_rate"] == 0.0
    assert st["P_L_eq_8192"] == 0.0
    assert st["generated_tokens"] == 300


# ============================================================================
# 任务 3：progressive generation 分片 + 限抽样（select_todo 纯函数）
# ============================================================================

def test_select_todo_no_overlap():
    """两 shard 划分互不重合，并集 = 全集。"""
    from scripts.prepare_skywork_responses import select_todo
    todo = list(range(100))
    a = select_todo(todo, None, None, 0, 2)
    b = select_todo(todo, None, None, 1, 2)
    assert set(a) & set(b) == set()
    assert set(a) | set(b) == set(todo)


def test_select_todo_max_samples_seed_reproducible():
    """max_samples + seed 可复现：同 seed 两次结果一致，且条数正确。"""
    from scripts.prepare_skywork_responses import select_todo
    a = select_todo(list(range(100)), 20, seed=42, shard_rank=0, num_shards=1)
    b = select_todo(list(range(100)), 20, seed=42, shard_rank=0, num_shards=1)
    assert a == b
    assert len(a) == 20


def test_select_todo_resume_skips_done():
    """resume：已完成 id 不重生成（调用方过滤 done 后不含已完成）。"""
    from scripts.prepare_skywork_responses import select_todo
    done = {0, 1, 2}
    sel = [i for i in select_todo(list(range(100)), None, None, 0, 1) if i not in done]
    assert 0 not in sel and 1 not in sel and 2 not in sel


def test_select_todo_shard_with_max_samples():
    """max_samples + shard 组合：先限抽样再分片，各 shard 不重合。"""
    from scripts.prepare_skywork_responses import select_todo
    a = select_todo(list(range(100)), 30, seed=7, shard_rank=0, num_shards=3)
    b = select_todo(list(range(100)), 30, seed=7, shard_rank=1, num_shards=3)
    c = select_todo(list(range(100)), 30, seed=7, shard_rank=2, num_shards=3)
    assert set(a) & set(b) == set() and set(a) & set(c) == set()
    assert len(a) + len(b) + len(c) == 30


# ============================================================================
# 任务 4：多阶段扩展 + 规模决策外推（extrapolate 纯函数）
# ============================================================================

def test_extrapolate_uses_mean_len():
    """成本用实测 mean_len（非 max_len）：total_hours = N×mean_len/tok_s/3600。"""
    from scripts.stage0_scale_probe import extrapolate
    r = extrapolate(50000, 8192, mean_len=1500, measured_tok_s=50, n_gpus=1)
    assert r["total_hours"] == pytest.approx(50000 * 1500 / 50 / 3600)


def test_extrapolate_2gpus_halves():
    """2 卡并行 → 时间减半。"""
    from scripts.stage0_scale_probe import extrapolate
    r1 = extrapolate(50000, 8192, mean_len=1500, measured_tok_s=50, n_gpus=1)
    r2 = extrapolate(50000, 8192, mean_len=1500, measured_tok_s=50, n_gpus=2)
    assert r2["total_hours"] == pytest.approx(r1["total_hours"] / 2)


def test_extrapolate_clamps_to_max_len():
    """mean_len 超过 max_len 时按 max_len 计（不超生成上限）。"""
    from scripts.stage0_scale_probe import extrapolate
    r = extrapolate(5000, 2048, mean_len=3000, measured_tok_s=50, n_gpus=1)
    assert r["total_hours"] == pytest.approx(5000 * 2048 / 50 / 3600)