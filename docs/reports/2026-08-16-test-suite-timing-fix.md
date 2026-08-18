# 全量测试耗时异常排查与修复报告

> 日期：2026-08-16 ｜ 状态：已修复并验证 ｜ 影响：全量 pytest 从 ~839s 恢复 ~78s

## 现象

- 此前（IMP-1b 之后）全量 `pytest tests/ -q` 耗时 ~67-78s（389 passed）。
- IMP-1c（repetition_penalty / loop_min_len）之后某次全量实测 ~839s（393 passed，-v 模式）。
- 单独跑任意文件/分组均快（每分组 11-70s），只有全量一起跑才慢——一度疑似测试交互/挂起。

## 排查过程（subagent，证据驱动）

1. `pytest tests/ -q --durations=25` 全量一次（EXIT=0），取 durations 定位最慢测试。
2. 分组隔离排除「单测试本身慢」：
   - test_scheduler + test_adaptive_cache + test_config = 62 passed 23s
   - test_l2_rollout / l2_budget / model_factory / l2_integration / pipeline = 114 passed 70s
   - test_experiment_stage3 / eval_aime / perf_equivalence = 45 passed 14s
   - 其余 143 passed 23s；test_losses + test_buffer = 22 passed 11s
3. benchmark `apply_repetition_penalty`（排除 IMP-1c 嫌疑）：
   - 默认 penalty=1.0 早退 = 0.0001 ms/次；
   - penalty>1 时 B=4/T=16 ≈ 0.24 ms、B=64/T=128 ≈ 4.5 ms/次。
   → **IMP-1c 彻底排除**（toy 规模整次 rollout 至多加几毫秒）。
4. 环境检查：.pytest_cache 仅 nodeids 26KB（不影响时长）；运行无残留文件；非 CPU/缓存问题。

## 根因

`test_stage0_teachers_hf_missing_ref_raises`（test_pipeline.py）走 `_stage0_teachers()` 的
hf 分支：先 `build_model(role="teacher")` → `HFCausalLM.from_pretrained("RL")` —— "RL" 非本地
路径 → transformers 尝试访问 HF hub（实测域名 hf-mirror.com）→ 网络不可达时阻塞等待超时
（7-50s；坏网络/重试下可放大到 ~839s）。之后才检查缺 `teacher_ref_path` 抛 ModelError。

## 修复（commit f08c13f）

1. `tests/conftest.py`：设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
   （所有 HF 相关测试均 mock from_pretrained 或假路径期待失败，离线后失败即返回，不触网）。
2. `test_pipeline.py::test_stage0_teachers_hf_missing_ref_raises`：mock `build_model` 返回
   占位，避免 from_pretrained 触网；仍验证「缺 teacher_ref_path → ModelError」被测行为。

## 结果

- 全量 **401 passed**（含 IMP-1c teacher 4 新测试），pytest 77.79s / wall 85.93s。
- 该网络测试从 7-50s（或 839s）降至 1s 内。
- 剩余 ~10s 为 torch CPU import 基线（本机固定开销，非网络）。

## 结论 / 建议

- 测试必须 hermetic：任何会触网的 `from_pretrained` 都要 mock 或设离线环境变量。
- 若 CI 复用本环境，建议保留 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 防回归。
- 全量耗时基线 = torch import ~10s + 393+ 测试 ~70s；任何超过该量级的变化都应先查触网测试。
