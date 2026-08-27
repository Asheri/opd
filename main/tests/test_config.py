"""config.py 单测：YAML 真加载、schema 校验（未知键/非法值报错）、点分覆盖、端到端可用。"""
from __future__ import annotations

import os

import pytest

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.exceptions import ConfigError
from fullstack_opd_v2.pipeline import DEFAULT_CONFIG_V2, FullStackOPDv2

YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "configs", "fullstack_opd.yaml")


def test_load_defaults_no_args():
    cfg = load_config()
    assert cfg["stage2"]["lr"] == DEFAULT_CONFIG_V2["stage2"]["lr"]
    assert cfg["vocab_size"] == 64
    # L1 默认翻转：学生 ref 一次性 rollout 拼胖 D（消曝光偏差）。schema 默认必须与
    # pipeline.DEFAULT_CONFIG_V2 同源，否则「schema 默认吞掉翻转」回归（P1-1）。
    assert cfg["stage1"]["warmup_source"] == "student_init"
    assert cfg["stage1"]["warmup_M"] == 4
    assert DEFAULT_CONFIG_V2["stage1"]["warmup_source"] == "student_init"
    assert DEFAULT_CONFIG_V2["stage1"]["warmup_M"] == 4


def test_load_real_yaml():
    cfg = load_config(path=YAML)
    assert cfg["stage1"]["cache_path"] == "fullstack_opd_cache_v2.pt"
    assert cfg["stage2"]["scheduling_mode"] == "fully_async"
    assert cfg["stage2"]["batch_size"] == 8


def test_unknown_top_level_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("vocab_size: 64\nbogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_unknown_nested_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage2:\n  n_steps: 5\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_bad_enum_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("dtype: fp16\n", encoding="utf-8")   # 非法枚举
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_dotted_overrides():
    cfg = load_config(overrides=["stage2.n_steps=50", "stage1.warmup_source=mix",
                                 "stage1.warmup_M=4", "top_k_student=256"])
    assert cfg["stage2"]["n_steps"] == 50
    assert cfg["stage1"]["warmup_source"] == "mix"
    assert cfg["stage1"]["warmup_M"] == 4
    assert cfg["top_k_student"] == 256


def test_override_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["stage2.nonexistent=1"])


def test_override_bad_value_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["stage2.rollout_engine=foo"])


def test_loaded_config_runs_end_to_end(tmp_path):
    """加载后的配置可直接喂给 FullStackOPDv2 跑通（集成校验）。"""
    cfg = load_config(overrides=[
        "n_prompts=8",
        "stage0.n_rl_steps=3",
        "stage2.n_steps=4",
        "stage2.batch_size=4",
        f"stage1.cache_path={tmp_path / 'c.pt'}",
    ])
    out = FullStackOPDv2(cfg, device="cpu").run()
    assert len(out["metrics"]) == 4


def test_new_sections_defaults():
    """T9：新增 run/logging/metrics/dataset/model_kind 段有默认值。"""
    cfg = load_config()
    assert cfg["model_kind"] == "toy"
    assert cfg["run"]["checkpoint_every"] == 10
    assert cfg["logging"]["level"] == "INFO"
    assert cfg["metrics"]["backend"] == "csv"
    assert cfg["dataset"]["type"] == "toy"


def test_new_sections_dotted_overrides():
    cfg = load_config(overrides=["run.checkpoint_every=5", "logging.level=DEBUG",
                                 "metrics.backend=wandb", "dataset.type=jsonl",
                                 "dataset.path=x.jsonl", "model_kind=toy"])
    assert cfg["run"]["checkpoint_every"] == 5
    assert cfg["logging"]["level"] == "DEBUG"
    assert cfg["metrics"]["backend"] == "wandb"
    assert cfg["dataset"]["path"] == "x.jsonl"


def test_new_section_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_bad_model_kind_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_kind: nope\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_deployment_keys_seeped_at_load():
    """顶层部署键应在 load_config 就按消费端分流注入 stage1/stage2（不再靠 pipeline 下渗）。"""
    cfg = load_config(overrides=[
        "dtype=bf16", "cache_mode=topk", "top_k_teacher=64",
        "top_k_student=64", "ref_topk=64", "offload_to_cpu=true"])
    assert cfg["stage1"]["cache_mode"] == "topk"
    assert cfg["stage1"]["top_k_teacher"] == 64
    assert cfg["stage2"]["dtype"] == "bf16"
    assert cfg["stage2"]["top_k_student"] == 64
    assert cfg["stage2"]["offload_to_cpu"] is True
    # ref_topk 保持纯顶层（pipeline 读顶层），不下渗到任何 stage
    assert cfg["ref_topk"] == 64
    assert "ref_topk" not in cfg["stage1"]
    assert "ref_topk" not in cfg["stage2"]


def test_snapshot_config_is_effective(tmp_path):
    """config.yaml 快照应等于有效运行时配置（下渗后）。"""
    from fullstack_opd_v2.run import RunManager
    cfg = load_config(overrides=["cache_mode=topk", "top_k_teacher=64"])
    paths = RunManager(cfg, run_dir=str(tmp_path / "r")).create()
    import yaml as _yaml
    with open(paths["config"], encoding="utf-8") as f:
        snap = _yaml.safe_load(f)
    assert snap["stage1"]["cache_mode"] == "topk"
    assert snap["stage1"]["top_k_teacher"] == 64


def test_stage2_ref_topk_not_seeped():
    """ref_topk 是顶层键，不注入 stage2（stage2 不接受该键，extra=forbid）。"""
    cfg = load_config(overrides=["ref_topk=128"])
    assert cfg["ref_topk"] == 128
    with pytest.raises(ConfigError):
        load_config(overrides=["stage2.ref_topk=128"])


def test_stage_subkey_priority_kept():
    """stage 显式子键优先于顶层下渗。"""
    cfg = load_config(overrides=["cache_mode=topk", "stage1.cache_mode=dense"])
    assert cfg["stage1"]["cache_mode"] == "dense"
    assert cfg["cache_mode"] == "topk"
    # stage2 无 cache_mode 槽位（死槽位已清，下渗已分流到 stage1）
    assert "cache_mode" not in cfg["stage2"]


def test_unimplemented_scheduling_mode_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage2:\n  scheduling_mode: n_step_off\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_validation_error_wrapped_as_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage2:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_seed_falls_back_to_top_level():
    """L4：run.seed 缺省时回退顶层 seed（不再被默认 42 遮蔽）。"""
    cfg = load_config(overrides=["seed=7"])
    assert cfg["run"]["seed"] is None
    assert cfg["seed"] == 7


def test_checkpoint_every_zero_rejected(tmp_path):
    """L5：checkpoint_every <= 0 抛 ConfigError（不再静默每步保存）。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("run:\n  checkpoint_every: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


# ---- L2 Adaptive Teacher Cache 配置（任务 1.1 / 6.1 前置）----

def test_l2_cfg_defaults_off():
    """L2Cfg 默认全关，l2.enabled=false 退回 L0/L1。"""
    from fullstack_opd_v2.config import load_config
    cfg = load_config()  # 无 path 用默认
    assert cfg["l2"]["enabled"] is False
    assert cfg["l2"]["cache"]["base_size"] == 50000
    assert cfg["l2"]["refresh_ratio"]["mode"] == "adaptive"
    assert cfg["l2"]["selective_rollout"]["enabled"] is True
    assert cfg["l2"]["disagreement"]["enabled"] is True
    assert cfg["l2"]["health_monitor"]["enabled"] is True


def test_l2_cfg_unknown_key_rejected():
    """extra=forbid：未知 l2 键报错。"""
    import pytest
    from fullstack_opd_v2.config import load_config, ConfigError
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.bogus=1"])


def test_dataset_apply_chat_template_override():
    """C3：dataset.apply_chat_template 可经 --set 覆盖（schema 声明，extra=forbid 不拒）。"""
    cfg = load_config(None, overrides=["dataset.apply_chat_template=true"])
    assert cfg["dataset"]["apply_chat_template"] is True
    cfg2 = load_config(None)
    assert cfg2["dataset"]["apply_chat_template"] is False


def test_l2_cfg_enable_via_override():
    """点分覆盖可开 L2 + 传子键（E0-E6 矩阵按此生成）。"""
    from fullstack_opd_v2.config import load_config
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=5", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=8"])
    assert cfg["l2"]["enabled"] is True
    assert cfg["l2"]["t_train"] == 5
    assert cfg["l2"]["m_refresh"] == 4
    assert cfg["l2"]["cache"]["refresh_size"] == 8
    assert cfg["l2"]["cache"]["max_response_length"] == 8


# ---- Stage 1 统一 K 存储架构（任务 S1-2：cache.top_k + cache.storage）----

def test_cache_cfg_defaults():
    """cache 段默认：top_k=32（稀疏下限）、storage=disk（本阶段目标）。"""
    cfg = load_config()
    assert cfg["cache"]["top_k"] == 32
    assert cfg["cache"]["storage"] == "disk"


def test_cache_cfg_topk_allowed():
    """top_k 允许 0（dense）/32/64/128/256（用户指定实验范围）。"""
    from fullstack_opd_v2.config import load_config
    for k in (0, 32, 64, 128, 256):
        assert load_config(overrides=[f"cache.top_k={k}"])["cache"]["top_k"] == k


def test_cache_cfg_storage_via_override():
    """storage 可切回 memory（显式保留原全量驻留路径）。"""
    cfg = load_config(overrides=["cache.storage=memory"])
    assert cfg["cache"]["storage"] == "memory"


def test_cache_cfg_rejects_bad_topk():
    """非 0/16/32/64/128/256 的 K 一律拒绝（固定 K，不做 adaptive K；16 自 2026-08-27 合法）。"""
    from fullstack_opd_v2.config import load_config
    for k in (1, 4, 8, 33):
        with pytest.raises(ConfigError):
            load_config(overrides=[f"cache.top_k={k}"])


def test_cache_cfg_rejects_bad_storage():
    """storage 仅 memory/disk。"""
    from fullstack_opd_v2.config import load_config
    with pytest.raises(ConfigError):
        load_config(overrides=["cache.storage=ram"])


def test_cache_cfg_unknown_subkey_rejected(tmp_path):
    """cache 段 extra=forbid：未知子键报错。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("cache:\n  top_k: 16\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))


def test_skywork_yaml_uses_cache_topk():
    """skywork_17b.yaml：cache.top_k=256（S1.5 K 校准定案）+ disk + stage2 无 top_k_student。"""
    cfg = load_config(path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "skywork_17b.yaml"))
    assert cfg["cache"]["top_k"] == 256
    assert cfg["cache"]["storage"] == "disk"
    # stage2 不再写死 top_k_student（yaml 已删，回落 0 → scheduler 用 cache.top_k=256）
    assert cfg["stage2"]["top_k_student"] == 0


def test_topk_16_allowed_paper_alignment():
    """Direct-OPD 论文对齐：cache.top_k=16 被 validator 接受（2026-08-27 新增）。"""
    cfg = load_config(overrides=["cache_mode=topk", "cache.top_k=16"])
    assert cfg["cache"]["top_k"] == 16


def test_topk_8_rejected():
    """cache.top_k=8 仍被 validator 拒绝（允许集 0/16/32/64/128/256）。"""
    import pytest
    with pytest.raises(Exception):
        load_config(overrides=["cache_mode=topk", "cache.top_k=8"])


def test_n_rollout_and_kl_adaptive_fields():
    """C2/C3 新配置字段：l2.rollout.n_rollout 与 stage2.kl_adaptive 可解析。"""
    cfg = load_config(overrides=[
        "l2.rollout.n_rollout=4", "stage2.kl_adaptive=true"])
    assert cfg["l2"]["rollout"]["n_rollout"] == 4
    assert cfg["stage2"]["kl_adaptive"] is True
    # 默认值（零回归）
    cfg2 = load_config()
    assert cfg2["l2"]["rollout"].get("n_rollout", 1) == 1
    assert cfg2["stage2"].get("kl_adaptive", False) is False
