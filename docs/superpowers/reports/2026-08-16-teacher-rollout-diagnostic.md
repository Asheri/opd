# IMP-1c：Teacher Rollout Capability（仅诊断/上界）报告

> 日期：2026-08-16 ｜ 状态：已实现并全绿（401 passed）｜ 性质：诊断能力，默认关闭

## 目标

为 L2 refresh rollout 增加可选的 **teacher 采样来源**（y ~ pi_teacher_rl），用于
diagnostic / upper-bound 实验：量化「若能得到教师质量 rollout，OPD 能提升多少」。
不改变主实验（student on-policy）路径；禁止默认启用、禁止混入主 E5。

## 设计

### config

`l2.rollout.rollout_source: Literal["student", "teacher"] = "student"`

- `student`（默认）：主实验路径 y ~ pi_student（现状零回归）
- `teacher`：仅诊断/上界 y ~ pi_teacher_rl；默认关闭 + pydantic Literal 拒绝非法值

### 数据流

```
l2.rollout.rollout_source=teacher
  → pipeline：读 config；teacher 时注入 teacher_rl.generate_with_status_kv（HF）
     或回落 run_refresh_phase 默认路径用 teacher_rl（toy）；打 logger.warning 诊断警告
  → run_refresh_phase：_gen 按 rollout_source 选生成模型（teacher→teacher_rl / student→student）
  → summary 记录 "source"（→ rollout/source 指标落盘 CSV）
  → ring buffer 逐样本 _source 元数据（append/get/state_dict/load_state_dict，兼容旧断点）
```

### 守卫

- 默认 student（禁止默认启用 teacher）
- pydantic Literal 拒绝非法值（如 `ref`）
- teacher 时 pipeline 显式 `logger.warning`：「诊断/上界实验专用，禁止混入主 E5；
  teacher 轨迹不构成主 on-policy 数据」
- E5 矩阵不设 teacher（默认 student）

## 测试（4 新增，全量 401 passed）

- config 默认/覆盖/非法值拒绝
- teacher source → 默认生成调用 teacher_rl（monkeypatch 捕获 `calls["model"] is t_rl`）
- student source → 默认生成调用 student
- ring buffer `_source` 元数据 state_dict 往返 + get 暴露

## GPU 验证要求（待办）

- 真实 teacher rollout 上界实验：`run_s2_real.py / run_l2_real.py
  --set l2.rollout.rollout_source=teacher` 与 student 对照，按 `rollout/source` 分组对比
  reward / valid rate / kl。
- 本机无 GPU，未伪造通过。

## 边界

- teacher 生成的样本显式启用时仍走 refresh 池，逐样本带 `source` 标记可追溯；
  若需严格隔离训练池，属后续实验设计决策（当前为诊断通道，默认不启用）。
