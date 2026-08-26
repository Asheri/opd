# OPD 项目文档索引

> 本文件是项目全部 Markdown 文档的导航入口（2026-08-25 整理后）。整理原则：分类归位、
> 已完成/过时进 `docs/archive/`、报告统一到 `docs/reports/`、去重、保留历史。所有路径相对仓库根。

## 1. 项目说明与规范（仓库根）

| 文件 | 定位 |
|---|---|
| `AGENTS.md` | 项目级规则**唯一权威**（语言/训练报错档案/GPU 并行/网络加速/架构/文档要求） |
| `CLAUDE.md` | **指针**：说明以 `AGENTS.md` 为准（保留文件名满足工具惯例，不单独维护） |
| `main/README.md` | `main/` 自研代码导览：论文抽取、代码地图、v1→v2 对照、审阅修复记录 |
| `main/fullstack_opd_v2/TECHNICAL_REPORT.md` | **技术文档与训练分析报告（唯一权威）**：工程实现/benchmark/显存/用时/数据构成/边界 |
| `docs/ISSUES_AND_FIXES.md` | 问题与漏洞分类总览（OOM/NCCL/配置/API 兼容等）→ 详细条目索引 |
| `DEPLOY.md` | 部署指南（GPU 服务 · 单一环境） |
| `OPTIMIZATION_PLAN_2xRTXPRO6000.md` | 当前硬件优化方案（2×RTX PRO 6000） |
| `RUNBOOK_2xPRO6000.md` | 上云运维手册（验收→数据→Stage0/1/2→监控→收敛→回滚） |

## 2. 设计文档（docs/specs/）

| 文件 | 定位 |
|---|---|
| `GPU_MEMORY_AND_PARALLEL_PLAN.md` | GPU 显存占用分析 + 并行训练方案（1.7B/4B/7B 三档） |
| `FULL_FLOW_OPTIMIZATION_MODEL.md` | 全流程数据流动层 + 优化目标数学模型 |
| `2026-08-08-gpu-perf-optimization-design.md` | GPU 性能优化设计（对应已归档实现计划） |
| `2026-08-09-eng-refactor-fixes-design.md` | 工程化二次审查修复设计（A/B/C/D） |
| `2026-08-13-align-directopd-experiment-design.md` | Direct-OPD 对齐实验设计 |
| `2026-08-14-adaptive-teacher-cache-design.md` | 自适应教师缓存设计（对应已归档实现计划） |

## 3. 进行中计划（docs/plans/）

- 当前无进行中计划（已完成的执行计划见 `docs/archive/plans/`）。

## 4. 报告（docs/reports/）

按日期专题报告（实现记录 / 诊断 / 校准 / 集成验证）。当前保留（均为权威/最新版）：
- `2026-08-14-adaptive-teacher-cache-implementation.md`（实现报告）
- `2026-08-15-budget-aware-eval.md`、`2026-08-15-budget-curve-analysis.md`（预算评估）
- `2026-08-15-stage0-scale-probe.md`、`2026-08-15-stage1-cache-layout.md`、`2026-08-15-stage1-cache-store.md`、`2026-08-15-stage1.5-k-calibration.md`（Stage 0/1 决策）
- `2026-08-15-stage2-rollout.md`（当前版，已完成服务器实测）
- `2026-08-16-teacher-rollout-diagnostic.md`、`2026-08-16-test-suite-timing-fix.md`
- `2026-08-17-imp1-imp3-commit-review.md`、`2026-08-17-imp1-rollout-loop-rootcause.md`、`2026-08-17-imp3-refresh-kl-anchor-correctness.md`、`2026-08-17-imp4-budget-aware-eval.md`、`2026-08-17-vllm-integration.md`（IMP 系列）
- `rollout_loop_calibration_chat.md`（模板口径最终校准 N=100，0/100 loop）
- 附件数据：`reports/budget_aware_data/`（budget 评估 JSONL/JSON + pilot 证据 csv/summary/decode + C1 验证日志）

> 旧版/被取代报告（stage2-rollout.legacy、旧 stage2-rollout、旧 N=48 校准、stale 裸 prompt 校准）已归档至 `docs/archive/reports/`。

## 5. 归档（docs/archive/，已完成/已过时/被取代）

- `*.md`（平铺，历史归档）：`2026-08-08-gpu-perf-optimization.md`、`2026-08-09-eng-refactor-fixes.md`、`2026-08-14-adaptive-teacher-cache.md`、`2026-08-14-skywork-1.7b-retrain-plan.md`、`OPTIMIZATION_PLAN_8xA100.md`、`OPTIMIZATION_PLAN_8x4090.md`
- `reports/`：被取代的旧报告（stage2-rollout 旧版/legacy、旧校准）
- `plans/`：已完成的执行计划（chat 模板三件套，含验收/阶段3结论/§9 决策/§10 C1 证据）
- `scripts/`：临时调试探针与一次性重跑脚本
- `requirements/`：已弃用的多环境依赖方案

每份归档首行均含"状态"标注；历史细节以 git 为准。

## 6. 错误档案

- **详细档案（唯一事实源）**：`C:\Users\12062\OneDrive\Desktop\items\training-errors.md`（全局规则指定路径，仓库外）。条目编号 E01-E45（历史项目）+ E1-E14（OPD v2/Skywork）。
- **项目内总览**：`docs/ISSUES_AND_FIXES.md`（按类别汇总+条目号索引，不复制细节）。

## 7. 其他文档

| 文件 | 定位 |
|---|---|
| `main/scripts/multistudent/README.md` | 多学生并发训练编排 |
| `benchmarks/aime24_25/README.md` | AIME24/25 蒸馏效果基准 |
| 上游 clone（`async-opd/`、`Direct-OPD/`、`Lightning-OPD/`） | 参照实现，文档不归本项目整理范围 |

## 8. 附录：docs/reports/ 全清单（当前）

- `docs/reports/2026-08-14-adaptive-teacher-cache-implementation.md`
- `docs/reports/2026-08-15-budget-aware-eval.md`
- `docs/reports/2026-08-15-budget-curve-analysis.md`
- `docs/reports/2026-08-15-stage0-scale-probe.md`
- `docs/reports/2026-08-15-stage1-cache-layout.md`
- `docs/reports/2026-08-15-stage1-cache-store.md`
- `docs/reports/2026-08-15-stage1.5-k-calibration.md`
- `docs/reports/2026-08-15-stage2-rollout.md`
- `docs/reports/2026-08-16-teacher-rollout-diagnostic.md`
- `docs/reports/2026-08-16-test-suite-timing-fix.md`
- `docs/reports/2026-08-17-imp1-imp3-commit-review.md`
- `docs/reports/2026-08-17-imp1-rollout-loop-rootcause.md`
- `docs/reports/2026-08-17-imp3-refresh-kl-anchor-correctness.md`
- `docs/reports/2026-08-17-imp4-budget-aware-eval.md`
- `docs/reports/2026-08-17-vllm-integration.md`
- `docs/reports/rollout_loop_calibration_chat.md`
- 证据数据：`docs/reports/budget_aware_data/`（MATH500 JSONL、pilot metrics/summary/decode、verify_c1_20260825.txt）

> 整理日志：2026-08-25 合并 `main/docs/superpowers/reports/` 至 `docs/reports/`（保留权威版、归档旧版）、
> 根目录 pilot 证据并入 `budget_aware_data/`、完成计划与临时脚本/requirements 归档、
> `CLAUDE.md` 改为 `AGENTS.md` 指针、删除 git 忽略缓存/零字节产物。核心业务代码未改动。
