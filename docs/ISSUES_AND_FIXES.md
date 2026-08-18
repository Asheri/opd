# 问题与漏洞总览（OPD v2 / Skywork 训练，含历史项目精选）

> 本文件按类别汇总本项目迄今遇到的全部问题与漏洞，每条给「症状一句话 + 当时措施 + 指向
> 详细档案条目」；**详细事实源 = `C:\Users\12062\OneDrive\Desktop\items\training-errors.md`**
> （编号 E01-E45 为历史项目；E1-E8、E9-E14 为 OPD v2/Skywork 阶段，本文重点覆盖后者）。
> 整理日期：2026-08-18。

## 1. 显存 OOM 类

| 问题 | 症状一句话 | 当时措施 | 档案 |
|---|---|---|---|
| cache build 真实词表 OOM | S2_E0 建缓存：dense (N,T,V=151936) 常驻致 80GB+，分配 11.59GiB 失败 | 降批/单行处理 + 缓存改 topk 稀疏；后续统一走 DiskTeacherCache | 2026-08-15 节 |
| **cli 建缓存 dense OOM** | `cli cache` 未注入 top_k/storage/pad_id，真实词表按 (N,T,V) 全量分配 | `_cmd_cache` 与 TrainPipeline 对齐注入 topk 稀疏参数 + 回归测试（commit d9c2ecb） | **E9** |
| **L2 显存预算假 OOM** | rollout 与训练同卡时 96GB 卡恒判不足，未按预算区分 | `_l2_rollout_mem_enough` 按同卡/异卡分别估算 + `_same_card`（commit 4e2f78f） | **E11** |
| **vLLM 引擎与训练共卡 OOM** | 引擎恒落 cuda:0，主进程 83-91GB + 引擎 11.6GB ≈ 94.9/94.97GB | 引擎按 rollout_device 映射 `CUDA_VISIBLE_DEVICES` 注入子进程（commit 93063b2） | **E12** |
| **staleness_q 队列 OOM（本轮根因）** | `_MIN_QUEUE_SIZE=16` 槽 × 在途 s_old (B,T,V) fp32 ≈ 85-95GB 常驻基线，batch 4/8 前向即撞顶 | `--set stage2.staleness_queue_min=2`（代码注释既有方案，注释见 scheduler.py）；候选：s_old 转 bf16/队列槽位收紧 | **E14** |
| 历史：DPO/RM fp32 权重 | policy+ref 双 fp32 94.94GB / 7B 默认 fp32 加载 | 8-bit 优化器 / 显式 bf16 / 模型量化 | E06/E12/E19/E40 |
| top-K 乱序截断（配套） | response_dists_topk 返回的 top-K 索引乱序，样本错位 | 修复排序 + 回归（IMP-2/P0） | E2 |

## 2. NCCL 权重同步类

| 问题 | 症状一句话 | 当时措施 | 档案 |
|---|---|---|---|
| vLLM 0.16 变更后首次同步部分生效 | 权重广播后部分层未刷新（P0 竞态） | C1 双层验证（ScatterGather + 扰动/复原门控 + 双发收敛），commit 9de57e0 | E3 |
| 扰动 lm_head 权重 logp 不变 | 扰动后生成 logp 差 = 0.000000，同步疑似失效 | 修正验证方法（全链路扰动 + 重新采样），确认 sync 生效 | E5 |
| ScatterGather 越界/device assert | 探针 gather 越界、device-side assert | 修复 gather 索引 + 单测（C1 探针） | E7 |
| 交叉分卡硬约束 | vLLM ≥0.16 WeightTransferEngine 与 trainer 同卡报 `Duplicate GPU detected` 卡死 | 布局固化：训练@GPU0 + vLLM@GPU1 互斥物理卡（AGENTS.md/CLAUDE.md 已记录） | 2026-08-17 |
| 引擎映射连带：import os 缺失 | 映射代码引用 os 未导入 → ModuleNotFoundError | rollout_vllm 补 `import os`（commit 2b5d384） | E13 |

## 3. 配置 / CLI / C2 守卫类

| 问题 | 症状一句话 | 当时措施 | 档案 |
|---|---|---|---|
| `apply_chat_template` schema 缺失 | pydantic `extra=forbid` 拒绝 `--set dataset.apply_chat_template` | DatasetCfg 显式声明该字段（commit 6608c5e） | C3 blocker |
| 顶层部署键被 stage 子字典静默忽略 | dtype/top_k/cache_mode 放顶层不生效（P0 型坑） | `load_config._seep_deployment_keys` 按消费端流下渗 stage1/stage2，新增键必须同步分流表 | CLAUDE.md 配置约定 |
| cli 建缓存 hashes 缺失 | 未传 tokenizer/教师一致性哈希 → C2 守卫拒绝加载 | `hash_models_from_cfg` 传递 + 校验（commit be4c011） | E10 |
| prepare 词表不匹配 | `--apply-chat-template` 后教师 tokenizer 词表守卫校验失败 | `prepare --force` 重生成 + 教师词表守卫（commit 38e89da） | C3/C1 |
| 缓存元数据不一致 | prompt_format/tokenizer_hash 与实际不符 → 训练静默错位风险 | cache 元数据显式记录 + C2 校验（topk 256 / prompt_format=chat / t_hash） | C2 |

## 4. API / transformers 兼容类（历史精选）

- E34 Auto 类移除（transformers 5.x）、E35 HfArgumentParser 重复字段、E36 `--bf16` 需值、
  E37 AWQ vs BitsAndBytesConfig 冲突、E41 `apply_chat_template` 返回 Tensor 非 dict（4.57.6 vs 5.x）、
  E42 async-opd 启动杀 GPU stale 进程（独占假设）——一行概括见下：
  **措施共性**：兼容写法 `tokenize=False + tokenizer()` 拿 dict；去重 dataclass 字段；量化模型不要传 BitsAndBytesConfig；GPU 任务与 async-opd 隔离。

## 5. 其他（数据/生成/流程）

| 问题 | 症状一句话 | 当时措施 | 档案 |
|---|---|---|---|
| rollout 75% loop + 乱码 token soup | 模板前 rollout 大量循环退化/生成乱码 | 根因定位（采样参数/pad_id/template）→ 2026-08-17-imp1-rootcause 报告 | E1 |
| 单卡串行违反 GPU 并行规则 | prepare 数据生成只用了单卡 | 按 AGENTS.md 双卡分片并行重生成（seed 确定性） | E8 |
| loop 检测旧校准 stale | 旧校准（裸 prompt，loop 率 75-87%）与模板 rollout 不符 | calibrate_rollout 增 `--chat/--temperature/--repetition-penalty` 重校准：48 条 0/48 loop → `loop_periods=[]` | 2026-08-18 |
| budget_eval 崩溃 | gt_extract 对 int answer 做 boxed 提取失败 | 修复提取兼容 | 2026-08-15 |

## 6. 代码修复提交索引（OPD v2 阶段，倒序）

- `2b5d384` import os 补漏（引擎映射连带）｜`93063b2` vLLM 引擎 CUDA_VISIBLE_DEVICES 映射
- `4e2f78f` L2 显存预算同卡/异卡｜`be4c011` cli hashes 传递｜`d9c2ecb` cli topk/storage/pad_id 注入
- `9de57e0` C1 vLLM 0.16 权重同步双层验证｜`38e89da` prepare --force + 教师词表守卫｜`6608c5e` DatasetCfg.apply_chat_template

## 7. 修复后运行状态快照（2026-08-18）

- 回归：本地 `443 passed`、服务器 `442 passed`。
- C1 权重同步三关全部 PASS（v5 全门控）。
- 缓存校验通过：`cache_skywork_chat.pt`（500 行 / topk 256 / max_response_len 2048 / prompt_format=chat / tokenizer_hash 一致）。
- 模板校准：`docs/reports/rollout_loop_calibration_chat.md`（0/48 loop）。
- **当前阻塞点**：Step 3 模板 pilot（E1/E2 20 步）——启动期 4 个 bug 已修复，staleness_q 队列显存（E14）为最近根因，已用 `stage2.staleness_queue_min=2` 收紧重试；若仍不足，候选 `stage2.batch_size=4` 或 s_old 入队前转 bf16（scheduler.py 显存注释同源）。
