# AIME 评估协议对齐说明（Direct-OPD 论文 vs 本项目）

> 日期：2026-08-13 ｜ 起因：用户质疑「论文 Qwen3-1.7B AIME24 48.3→58.3、Qwen3-4B 72.5→77.6，而本项目只有 6.7→13.3 / 16.7」
> 以及「17B 中间断点 step80=3.3% → step120=13.3% → step160=3.3% 非单调，怀疑数据有问题」。

## 结论一：分数数量级差异 = 评估协议不同，不是蒸馏失败

**Direct-OPD 论文（arXiv:2607.05394）附录 A Table 2 的评估协议：**

| 评估设置 | 论文 | 本项目（此前） |
|---------|------|--------------|
| Samples per problem | **32** | 1 |
| Sampling temperature | **0.7** | 0.0（贪心） |
| Top-p | **0.95** | 无 |
| Maximum generation length | **31,744** | 2,048 |
| 指标 | **ave@32**（每题 32 采样平均正确率） | pass@1（贪心单样本） |

- 论文所有分数（图 1/2、Table 1）纵轴均标 `ave@32`；launch 脚本 `train_justrl_qwen.sh`
  `VAL_N=32`、`val_kwargs.temperature=0.7`、`val_kwargs.top_p=0.95`、`val_kwargs.do_sample=True`。
- ave@32 的 32 次采样让每题正确率从「0/1 二元」变成「0~1 连续」，小模型在 AIME 难题上显著高于贪心——
  这是 48.3 vs 6.7 差一个数量级的主因。**两个数字不可直接比较。**

## 结论二：17B 断点非单调（3.3→13.3→3.3）= 30 题小样本 + 贪心的噪声，非数据错误

- 30 题下 **3.3% = 1/30、13.3% = 4/30**——step80 与 step160 都只答对 1 题，step120 答对 4 题，差距仅 3 题。
- 贪心解码下每题非 0 即 1，30 题方差极大；这正说明为何论文用 ave@32（方差降低一个量级）。
- **但**：当前评估不足以确证 step120 是真正最优断点——须用论文同款 ave@32 协议重验。

## 修复：eval-aime 已支持 ave@32 对齐协议（commit 2b0b268）

`eval-aime` 新增三个参数（本地 22 测试已绿）：

```bash
python -m fullstack_opd_v2 eval-aime \
  --model <HF 模型路径> \
  --datasets AIME24 AIME25 \
  --n-samples 32 --temperature 0.7 --top-p 0.95 \
  --metric ave --prompt-style dapo \
  --out <输出目录> --device cuda:N
```

- `--metric ave`：accuracy = 每题 32 采样中答对比例的均值（论文 ave@32 口径）。
  `--metric pass1`（默认）保持原行为（任一采样对即对），向后兼容。
- `--top-p 0.95`：采样时透传到 model.generate（论文协议）。
- `--prompt-style dapo`：Direct-OPD 论文附录 A 的 DAPO 模板（"Answer:" 结尾行），
  `--prompt-style boxed`（默认）保持原 \boxed{} 模板。
- CLI 覆盖优先于 run-dir config.yaml 的 eval.* 配置。

## 待办：服务器恢复后重评估

用 ave@32 协议重跑以下检查点，验证 step120 最优性并尝试对齐论文数字：

| 模型 | 目的 |
|------|------|
| 1.7B 基座（Qwen3-1.7B） | 对齐论文 48.3 起点（若协议一致应接近） |
| 1.7B ms_step120 | 验证是否真最优（论文口径） |
| 1.7B ms_step80 / ms_step160 | 验证 3.3% 是否仍是噪声（ave@32 应更平滑） |
| 4B v3_step59 | 对齐论文 4B 72.5→77.6（协议一致应接近 70+） |

**注意事项**：
- 论文用 Skywork-OR1 math 数据 + DAPO 模板训练与评估；本项目用 GSM8K + boxed 模板——
  即便协议相同，**训练数据/模板不同会导致绝对分数与论文仍有差距**，预期是"量级对齐"而非"逐点相等"。
- ave@32（每题 32 次采样）评估耗时约为贪心的 ~32×（batch 内可并行，实际约 8~15× 墙钟）。
  三档学生 + 断点可在 2×RTX PRO 6000 双卡并行跑。
