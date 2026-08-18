# OPD 项目文档索引

> 本文件是项目全部 Markdown 文档的导航入口（2026-08-18 整理后）。整理原则：分类归位、
> 已完成/过时进 `docs/archive/`、去 `superpowers` 冗余层、重复文件保留并标注关系、
> 不重写既有正文。所有路径相对仓库根。

## 1. 项目说明与规范（仓库根）

| 文件 | 定位 |
|---|---|
| `AGENTS.md` | 项目级智能体规范入口（与 `CLAUDE.md` 内容完全相同，用户要求双份保留）。**以 `AGENTS.md` 为最新口径** |
| `CLAUDE.md` | 同上（保留给其他工具/习惯用法访问；如两者冲突以 AGENTS.md 为准） |
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

- `2026-08-18-chat-template-three-piece-execution.md` —— chat 模板三件套执行计划（Step0-3 + 验收表，**进行中**）

## 4. 报告（docs/reports/）

按日期专题报告（实现记录 / 诊断 / 校准 / 集成验证）。**旧版标注**：
- `2026-08-15-stage2-rollout.md` —— 当前版（08-16 更新）
- `2026-08-15-stage2-rollout.legacy.md` —— 旧版（源自 main/docs，保留备查）
- 最近关键报告：`2026-08-17-vllm-integration.md`（vLLM 接入+C1 验证）、`2026-08-17-imp1-rollout-loop-rootcause.md`（loop 根因）、`2026-08-16-rollout-loop-calibration.md`（旧校准）、`rollout_loop_calibration_chat.md`（模板口径重校准 0/48）、`2026-08-14-adaptive-teacher-cache-implementation.md`
- 附件数据：`reports/budget_aware_data/`（budget 评估 JSONL/JSON）

完整清单见下文 `## 7. 附录：reports/ 全清单`。

## 5. 归档（docs/archive/，已完成/已过时）

- `2026-08-08-gpu-perf-optimization.md`（已完成）
- `2026-08-09-eng-refactor-fixes.md`（已完成）
- `2026-08-14-adaptive-teacher-cache.md`（已完成）
- `2026-08-14-skywork-1.7b-retrain-plan.md`（已执行：实际 2048 双卡口径）
- `OPTIMIZATION_PLAN_8xA100.md`（8×A100 方案，已过时）
- `OPTIMIZATION_PLAN_8x4090.md`（8×4090 方案，已过时）

每份首行均加"状态"标注；历史细节以 git 为准。

## 6. 错误档案

- **详细档案（唯一事实源）**：`C:\Users\12062\OneDrive\Desktop\items\training-errors.md`（全局规则指定路径，仓库外）。条目编号 E01-E45（历史项目）+ E1-E8 / E9-E14（OPD v2/Skywork）。
- **项目内总览**：`docs/ISSUES_AND_FIXES.md`（按类别汇总+条目号索引，不复制细节）。

## 7. 其他文档

| 文件 | 定位 |
|---|---|
| `main/scripts/multistudent/README.md` | 多学生并发训练编排 |
| `benchmarks/aime24_25/README.md` | AIME24/25 蒸馏效果基准 |
| `.workbuddy/memory/2026-08-{05..08}.md` | 会话决策日志（持续追加） |
| 上游 clone（`async-opd/`、`Direct-OPD/`、`Lightning-OPD/`） | 参照实现，文档不归本项目整理范围 |

## 8. 附录：docs/reports/ 全清单

- `docs/reports/2026-08-14-adaptive-teacher-cache-implementation.md`
- `docs/reports/2026-08-15-budget-aware-eval.md`
- `docs/reports/2026-08-15-budget-curve-analysis.md`
- `docs/reports/2026-08-15-stage0-scale-probe.md`
- `docs/reports/2026-08-15-stage1-cache-layout.md`
- `docs/reports/2026-08-15-stage1-cache-store.md`
- `docs/reports/2026-08-15-stage1.5-k-calibration.md`
- `docs/reports/2026-08-15-stage2-rollout.legacy.md`
- `docs/reports/2026-08-15-stage2-rollout.md`
- `docs/reports/2026-08-16-rollout-loop-calibration.md`
- `docs/reports/2026-08-16-teacher-rollout-diagnostic.md`
- `docs/reports/2026-08-16-test-suite-timing-fix.md`
- `docs/reports/2026-08-17-imp1-imp3-commit-review.md`
- `docs/reports/2026-08-17-imp1-rollout-loop-rootcause.md`
- `docs/reports/2026-08-17-imp3-refresh-kl-anchor-correctness.md`
- `docs/reports/2026-08-17-imp4-budget-aware-eval.md`
- `docs/reports/2026-08-17-vllm-integration.md`
- `docs/reports/rollout_loop_calibration_chat.md`

> 整理日志：2026-08-18 执行 `docs/superpowers/{plans,specs,reports}` → `docs/{plans,specs,reports,archive}` 提升；
> `main/docs/` 两份报告合并入 `docs/reports/`（旧版加 `.legacy` 后缀）；旧路径引用已在被移动文件内统一替换为 `docs/*`。
> 唯一保留的旧路径引用：`main/fullstack_opd_v2/experiment.py` 一处代码注释（开发提示，非功能路径）。
