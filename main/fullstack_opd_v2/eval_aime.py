"""AIME 评估器（main/ 一等能力，替代原 benchmark 对 async-opd opd.cli.eval 的依赖）。

main/ 是真正主项目：真实模型（HF 权重）在 AIME24/AIME25 上的评估在此自包含实现——
模型加载走 transformers（AutoModelForCausalLM + AutoTokenizer，本地路径 / HF id），
数据集走 huggingface datasets，答案提取用 \boxed{} 级联 → 整数，评分精确匹配。

- `extract_answer(text)` / `normalize_answer(a)`：纯函数，无模型依赖，可单测。
- `AimeEvaluator`：真实评估器（model load + dataset + generate + score + jsonl 落盘）。
- CLI 入口见 cli.py 的 `eval-aime` 子命令；`--run-dir` 桥接读 run_dir/config.yaml
  的 `eval.*` 配置（model_path / max_new_tokens / n_samples / temperature）。
"""

from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass

from .exceptions import ConfigError, DataError, ModelError, TrainingError

# AIME 数据集别名 → HF dataset 名（列：problem + answer，答案整数）
AIME_DATASETS = {
    "AIME24": "Maxwell-Jia/AIME_2024",
    "AIME25": "yentinglin/aime_2025",
}
DEFAULT_DATASETS = ("AIME24", "AIME25")
# 生成侧上下文上限（prompt + max_new 之和不得超过；与 transformers 默认 4096 对齐）
_MAX_CONTEXT = 4096

# 标准推理提示（答案必须放 \boxed{}）
PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
# DAPO 风格模板（Direct-OPD 论文附录 A：训练 rollouts 与评估 prompts 同用）——
# 要求最后一行 "Answer: <数字>"。对齐论文评估协议时用 prompt_style="dapo"。
PROMPT_TEMPLATE_DAPO = (
    "Solve the following math problem step by step.\n"
    "The last line of your response should be of the form\n"
    "Answer:\n"
    "$Answer (without quotes) where $Answer is the answer to the problem.\n"
    "{problem}\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)


# --------------------------- 纯函数（可单测） ---------------------------
def format_prompt(problem: str, style: str = "boxed") -> str:
    """把 AIME 题目格式化为推理 prompt。

    style="boxed"（默认）→ \boxed{} 模板；style="dapo" → Direct-OPD 论文附录 A 的
    DAPO 模板（"Answer:" 结尾行），对齐论文评估协议。
    """
    tpl = PROMPT_TEMPLATE_DAPO if style == "dapo" else PROMPT_TEMPLATE
    return tpl.format(problem=str(problem).strip())


def extract_answer(text: str, style: str = "boxed") -> str:
    """从模型输出提取数值答案。

    style="dapo"：优先取 "Answer:" 行后的数字（论文 DAPO 模板的落点）；
    否则级联 \boxed{...} → 其中第一个数字；再回退最后一个数字。
    返回原始字符串（含可能的负号/千分位），未找到返回 ""。
    """
    if style == "dapo":
        # 模板契约要求答案在【最后一行】"Answer:"（PROMPT_TEMPLATE_DAPO）。
        # 长 CoT 可能中途写下 "answer:" 草稿再修正——取最后一个匹配（P2 修复，
        # 否则首个匹配命中草稿 → 本应判对的采样被判错，系统性低估 ave@32）。
        matches = re.findall(r"[Aa]nswer\s*:\s*([^\n]+)", text)
        if matches:
            mm = re.search(r"-?\d[\d,]*", matches[-1])
            if mm:
                return mm.group(0)
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        for cand in reversed(boxed):
            m = re.search(r"-?\d[\d,]*", cand)
            if m:
                return m.group(0)
        return boxed[-1].strip()
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1] if nums else ""


def normalize_answer(a) -> int | None:
    """把答案规范化成整数（AIME 答案为 3 位整数，005 与 5 等价）。"""
    if a is None:
        return None
    s = str(a).strip().replace(",", "")
    if not re.fullmatch(r"-?\d+", s):
        return None
    return int(s)


# --------------------------- 论文评分（Direct-OPD ttrl_math fast 路径） -------------
# 对齐论文 `reward_score/ttrl_math` 的默认 `grade()`（fast=True）：
#   `grade_answer_mathd(...) or grade_answer_sympy(...)`。
# 与简单整数匹配不同，它做数学等价判定（3/4 与 0.75、\frac{25}{8} 等价等），
# 且提取用「最后一个 \boxed{}」级联（长 CoT 中途草稿不污染）。
# 依赖 sympy（论文运行环境必装）；缺失时报错而非静默降级回整数匹配。

def _boxed_last(s: str) -> str | None:
    """取最后一个 \\boxed{...} 的完整文本（用花括号配对找闭合，论文 last_boxed_only_string）。"""
    s = s.replace("\\fbox", "\\boxed")
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    left = 0
    while i < len(s):
        if s[i] == "{":
            left += 1
        elif s[i] == "}":
            left -= 1
            if left == 0:
                return s[idx:i + 1]
        i += 1
    return None


def _remove_boxed(s: str) -> str | None:
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return None


def _extract_boxed_answer(text: str) -> str | None:
    """论文 extract_boxed_answer：取最后一个 \\boxed{} 内容。"""
    b = _boxed_last(text)
    if b is None:
        return None
    return _remove_boxed(b)


def _parse_latex(expr: str) -> str:
    """论文 _parse_latex：pylatexenc 把 latex 转成 sympy 可读的普通文本。

    - \frac -> " \frac"（混合数友好）→ latex2text 输出如 "25/8"；
    - 常见数学符号（√ π ∞ ∪ · ×）替换为文本等价。
    依赖 pylatexenc（论文运行环境必装）；缺失时原样返回（宁可少判对也不报错）。
    """
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        expr = LatexNodes2Text().latex_to_text(expr)
    except Exception:
        return expr
    expr = (expr.replace("√", "sqrt").replace("π", "pi")
            .replace("∞", "inf").replace("∪", "U")
            .replace("·", "*").replace("×", "*"))
    return expr.strip()


def _norm_sympy(expr: str) -> str:
    """论文 _normalize（对称归一化，去单位/空格/%/latex → sympy 可读）。量级对齐原实现。"""
    if expr is None:
        return ""
    expr = str(expr)
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = (expr.replace("\\%", "%").replace("\\$", "$")
            .replace("$", "").replace("%", ""))
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in ["degree", "cm", "centimeter", "meter", "mile", "second", "minute",
                 "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard"]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(r",\\! *", "", expr)
    try:
        if float(expr) and abs(float(expr) - round(float(expr))) <= 1e-7:
            expr = str(int(round(float(expr))))
    except Exception:
        pass
    # 论文顺序：含 latex 时先转文本（\frac{25}{8} -> 25/8），再做混合数/去空格/去 {}。
    if "\\" in expr:
        expr = _parse_latex(expr)
    expr = re.sub(r"- *", "-", expr)
    # 混合数 7 3/4 -> 7+3/4
    expr = re.sub(r"(\d) +(\d)", r"\1+\2", expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.lower()
    try:
        f = float(expr)
        if abs(f - round(f)) <= 1e-7:
            expr = str(int(round(f)))
    except Exception:
        pass
    return expr


def _sympy_parse(expr: str):      # pragma: no cover - 依赖 sympy
    """论文 _sympy_parse：^ -> **，implicit multiplication。"""
    import sympy
    from sympy.parsing import sympy_parser
    py = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py, transformations=(sympy_parser.standard_transformations
                              + (sympy_parser.implicit_multiplication_application,)))


def _are_equal_under_sympy(gt: str, given: str) -> bool:    # pragma: no cover
    """论文 are_equal_under_sympy：sympy 化简差为 0 判等。"""
    import sympy
    bad = ["^{", "^(", re.compile(r"\^[0-9]+\^"), re.compile(r"\^[0-9][0-9]+")]
    expr = f"({gt})-({given})"
    if any((b in expr if isinstance(b, str) else b.search(expr)) for b in bad):
        return False
    if len(set(c for c in expr if c.isalpha() and c not in "sqrfrac")) > 2:
        return False
    try:
        diff = _sympy_parse(expr)
        return sympy.simplify(diff) == 0
    except Exception:
        return False


def _grade_answer_sympy(given: str, ground_truth: str) -> bool:    # pragma: no cover
    """论文 grade_answer_sympy（fast 路径，无 math_verify 兜底）。"""
    gt_n, gv_n = _norm_sympy(ground_truth), _norm_sympy(given)
    if gt_n == gv_n:
        return True
    if not gv_n:
        return False
    if gt_n == given and len(given) == 0:
        return False
    # 单元素（AIME 答案都是整数/分数，忽略 tuple 分支）
    f_gt = re.fullmatch(r"-?\d+/\d+", gt_n)
    f_gv = re.fullmatch(r"-?\d+/\d+", gv_n)
    if f_gt and f_gv:
        return gt_n == gv_n
    i_gt = re.fullmatch(r"-?\d+", gt_n)
    i_gv = re.fullmatch(r"-?\d+", gv_n)
    if bool(i_gt) != bool(i_gv):
        return False
    return _are_equal_under_sympy(gt_n, gv_n)


def _grade_answer_mathd(given: str, ground_truth: str) -> bool:
    """论文 grade_answer_mathd：mathd 归一化后字符串相等。"""
    return _norm_sympy(given) == _norm_sympy(ground_truth)


def _validate_device(device: str) -> str:
    """校验 device 格式（P2 修复）：合法为 cpu | cuda[:N]。

    传 '0'（应为 'cuda:0'）此前抛误导性 ModelError（"路径/HF id 无效"）；
    前置校验给明确错误，避免排查方向被带偏。
    """
    s = str(device).strip().lower()
    if s == "cpu":
        return s
    if s == "cuda" or re.fullmatch(r"cuda:\d+", s):
        return s
    raise ConfigError(
        f"device={device!r} 非法：须 'cpu' 或 'cuda[:N]'（如 'cuda:0'）；"
        "裸数字（如 '0'）不是合法 device")


# --------------------------- 评估器 ---------------------------
@dataclass
class AimeResult:
    dataset: str
    model_path: str
    correct: int
    total: int
    rows: list[dict]
    # metric=ave 时：每题 n 采样中答对比例的均值（对齐论文 ave@32 口径）。
    # None → pass@1 口径（correct/total）。
    ave_accuracy: float | None = None

    @property
    def accuracy(self) -> float:
        if self.ave_accuracy is not None:
            return self.ave_accuracy
        return self.correct / self.total if self.total else 0.0

    @classmethod
    def ave32_from_rows(cls, rows: list[dict]) -> float | None:
        """从落盘 rows（含 correct_count/n_samples）重算 ave@32（论文口径）。

        对每题算 correct_count/n_samples（正确比例），30 题平均。用于审计/复现：
        与 evaluate() 的 ave_accuracy 应逐位一致。缺采样级字段返回 None。
        """
        fracs = []
        for r in rows:
            cc = r.get("correct_count")
            ns = r.get("n_samples")
            if cc is None or not ns:
                return None
            fracs.append(cc / ns)
        return (sum(fracs) / len(fracs)) if fracs else None


class AimeEvaluator:
    """真实模型 AIME 评估器（transformers 后端）。

    n_samples>1 时对每题采样 N 条，`correct` 记 pass@1（任一采样答对即对）。
    """

    # 类级默认：测试用 object.__new__ 绕过 __init__ 构造时也具该属性（不设则 evaluate 报错）。
    scoring = "int"
    chat_template = False

    def __init__(self, model_path: str, device: str = "cpu",
                 max_new_tokens: int = 2048, batch_size: int = 8,
                 n_samples: int = 1, temperature: float = 0.0,
                 trust_remote_code: bool = False, dtype: str = "auto",
                 top_p: float | None = None,
                 metric: str = "pass1",
                 prompt_style: str = "boxed",
                 scoring: str = "int",
                 chat_template: bool = False):
        # P2：参数校验前置（transformers 导入/模型加载之前），配置错快速失败、零副作用。
        # 上下文上限按模型 config 动态取（Qwen3=40960，对齐论文 MAX_VAL_RESP_LENGTH 31744）；
        # 模型加载后才得知，故保守前置校验用 _MAX_CONTEXT（历史默认 4096）挡明显非法值，
        # 模型加载后 _resolve_max_context 按实际 config 复核（可放宽）。
        if int(max_new_tokens) >= _MAX_CONTEXT and int(max_new_tokens) > 32768:
            raise ConfigError(
                f"max_new_tokens={max_new_tokens} 异常大（>32768）；请检查")
        if scoring not in ("int", "sympy"):
            raise ConfigError(f"scoring={scoring!r} 非法：须 int | sympy（sympy=论文数学等价判定）")
        self.scoring = scoring
        # chat_template=True：对齐论文 verl 验证（schemas.py _handle_apply_chat_template）——
        # 用模型 chat template 包裹 prompt（<|im_start|>user/assistant），非裸字符串。
        self.chat_template = bool(chat_template)
        self.n_samples = max(1, int(n_samples))
        self.temperature = float(temperature)
        self.top_p = float(top_p) if top_p is not None else None
        if metric not in ("pass1", "ave"):
            raise ConfigError(f"metric={metric!r} 非法：须 pass1 | ave（ave=论文 ave@32 平均正确率）")
        self.metric = metric
        if prompt_style not in ("boxed", "dapo"):
            raise ConfigError(f"prompt_style={prompt_style!r} 非法：须 boxed | dapo（论文模板）")
        self.prompt_style = prompt_style
        # P2（R2 审查）：greedy + 多采样会退化为 num_return_sequences>1 的重复/崩溃
        # （temperature<=0 → do_sample=False → 同种子多序列逐字重复，pass@1 被污染）。
        # 采样模式由温度驱动：n>1 且 T>0 才合法；要么 T>0（真采样），要么 n==1（贪心）。
        if self.n_samples > 1 and self.temperature <= 0:
            raise ConfigError(
                f"n_samples={self.n_samples}>1 但 temperature={self.temperature}<=0："
                "贪心解码下多序列逐字重复，pass@1 无意义。请设 temperature>0 或 n_samples=1")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:                       # pragma: no cover
            raise ModelError(f"AIME 评估需要 transformers：{e}") from e
        self.model_path = model_path
        # P2（二次审阅修复）：device 前置校验——'0' 而非 'cuda:0' 会抛误导性
        # ModelError（"路径/HF id 无效"）。合法：cpu | cuda[:N]。
        self.device = _validate_device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))
        # 上下文上限：模型 config 的 max_position_embeddings（Qwen3=40960，对齐论文长生成）。
        # 模型加载后从 config 取；缺省回退 _MAX_CONTEXT。
        self.max_ctx = _MAX_CONTEXT
        try:
            self.tok = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            raise ModelError(
                f"加载 tokenizer {model_path!r} 失败（路径/HF id 无效？）：{e}") from e
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # P1（二次审阅修复）：decoder-only 批量生成必须左填充（右填充 → RoPE 位置被
        # pad 位移 → 生成 token 质量退化）。transformers 对右填充只打 warning 不自动修。
        # 注意 tokenizer 实例可被复用，显式设置避免副作用。
        self.tok.padding_side = "left"
        # dtype：显式 bf16/float16 用对应精度；'auto' 在 CUDA 上默认 bf16（现代卡），CPU 用 fp32。
        if dtype in ("bf16", "float16"):
            torch_dtype = {"bf16": "bfloat16", "float16": "float16"}[dtype]
        elif dtype == "auto" and str(device).startswith("cuda"):
            torch_dtype = "bfloat16"
        else:
            torch_dtype = None
        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **kwargs).to(device).eval()
        except Exception as e:
            raise ModelError(
                f"加载模型 {model_path!r} 失败（路径/HF id 无效？）：{e}") from e
        # 模型加载后按真实 config 复核上下文上限（可放宽到 max_position_embeddings，
        # 对齐论文 MAX_VAL_RESP_LENGTH=31744；缺省回退 4096）。
        # ⚠️ 只接受真实 int（mock/None 跳过）：测试的 fake model config 是 Mock 对象。
        mpe = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(mpe, int) and mpe > 1:
            self.max_ctx = mpe
        if self.max_new_tokens >= self.max_ctx:
            raise ConfigError(
                f"max_new_tokens={self.max_new_tokens} ≥ 模型上下文上限 {self.max_ctx}；请调小")
        # P2（二次审阅修复 #5）：max_new 逼近 max_ctx 时 prompt 被静默截到近零长度
        # （max_length=max(1, max_ctx-max_new) → 题目信息全丢）。预留 ≥20% 上下文给 prompt。
        if self.max_new_tokens > 0.8 * self.max_ctx:
            raise ConfigError(
                f"max_new_tokens={self.max_new_tokens} 超过上下文上限 {self.max_ctx} 的 80%："
                f"生成太长会把 prompt 截断（题目信息丢失）。请调小 max_new_tokens 或换更大上下文模型")

    def close(self):
        """释放模型/tokenizer 与 GPU 显存。"""
        if hasattr(self, "model"):
            try:
                self.model.to("cpu")
            except Exception:
                pass
            del self.model
        if hasattr(self, "tok"):
            del self.tok
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --------------------------- 数据 ---------------------------
    def resolve_dataset(self, dataset_ref: str) -> str:
        return AIME_DATASETS.get(dataset_ref, dataset_ref)

    def load_problems(self, dataset_ref: str) -> list[tuple[str, str]]:
        """加载 AIME 题目 → [(problem, answer)]。"""
        try:
            from datasets import load_dataset
        except Exception as e:                       # pragma: no cover
            raise ModelError(f"AIME 评估需要 datasets：{e}") from e
        name = self.resolve_dataset(dataset_ref)
        try:
            ds = load_dataset(name, split="train")
        except Exception as e:
            raise DataError(f"加载 AIME 数据集 {name!r} 失败：{e}") from e
        # 列名兼容：不同 AIME 数据集列名不一（problem/Problem/Question/Prompt、
        # answer/Answer/Solution）——大小写不敏感 + 候选键匹配（部署实测
        # Maxwell-Jia/AIME_2024 是大写 Problem/Solution/Answer）。
        prob_keys = ("problem", "question", "prompt", "Problem", "Question", "Prompt")
        ans_keys = ("answer", "solution", "Answer", "Solution")
        rows = []
        for row in ds:
            prob = next((row.get(k) for k in prob_keys if row.get(k) is not None), None)
            ans = next((row.get(k) for k in ans_keys if row.get(k) is not None), None)
            if prob is None or ans is None:
                raise DataError(
                    f"数据集 {name!r} 缺 problem/answer 列（实际列：{list(row.keys())}）")
            rows.append((str(prob), str(ans)))
        if not rows:
            raise DataError(f"数据集 {name!r} 为空")
        return rows

    # --------------------------- 生成 ---------------------------
    def generate(self, prompts: list[str]) -> list[str]:
        """批量贪心/采样生成；每个 prompt 产出 n_samples 条响应（拍平返回）。

        批量 tokenize + 一次 model.generate（num_return_sequences=n_samples），
        避免逐 prompt 多次模型调用（R1 性能修复）。
        """
        import torch
        import math
        responses: list[str] = []
        n = self.n_samples
        do_sample = self.temperature > 0 and n > 1
        # P2（R2 审查）+ 二次审阅修复 #6（实质修复）：num_return_sequences=n 把每批序列数
        # 放大 n 倍，峰值 KV/显存随 batch×n 线性涨（部署实测：n=32 × 30000 token 长生成
        # 在 4B 上 OOM——93GB 撑爆 96GB）。真正解决：把每批的序列数压到 batch_size 量级——
        #   - 每批 prompt 数 = batch_size // chunk（prompt 维度收窄）
        #   - 每个 prompt 的 n 条采样拆成 ceil(n/chunk) 个子批（num_return_sequences=chunk），
        #     峰值序列数 = chunk × (batch_size//chunk) ≈ batch_size，与 n 无关。
        chunk = n if n <= 1 else max(1, min(self.batch_size, n))
        step = self.batch_size if n <= 1 else max(1, self.batch_size // chunk)
        n_chunks = max(1, math.ceil(n / chunk)) if n > 1 else 1
        try:
            for i in range(0, len(prompts), step):
                batch = prompts[i:i + step]
                if self.chat_template:
                    # 对齐论文 verl 验证：apply_chat_template 包裹（<|im_start|>user/assistant）。
                    # prompt 已是 boxed 模板文本（含"reason step by step...\boxed{}"），
                    # 作为 user 消息传入；add_generation_prompt=True 追加 assistant 头。
                    batch = [self.tok.apply_chat_template(
                        [{"role": "user", "content": p}],
                        add_generation_prompt=True, tokenize=False)
                        for p in batch]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True,
                               max_length=max(1, self.max_ctx - self.max_new_tokens))
                enc = {k: v.to(self.device) for k, v in enc.items()}
                seq_len = enc["input_ids"].size(1)
                for _c in range(n_chunks):
                    with torch.no_grad():
                        gen_kwargs = dict(
                            max_new_tokens=self.max_new_tokens,
                            do_sample=do_sample, num_return_sequences=chunk,
                            temperature=self.temperature if do_sample else 1.0,
                            pad_token_id=self.tok.pad_token_id)
                        if do_sample and self.top_p is not None:
                            gen_kwargs["top_p"] = self.top_p
                        out = self.model.generate(**enc, **gen_kwargs)
                    for o in out:
                        responses.append(self.tok.decode(o[seq_len:],
                                                         skip_special_tokens=True))
        except Exception as e:
            raise TrainingError(f"AIME 生成失败：{e}") from e
        return responses

    # --------------------------- 评估 ---------------------------
    def _grade_sympy(self, pred: str, gt: str) -> bool:
        """论文 grade()：grade_answer_mathd(...) or grade_answer_sympy(...)。"""
        if not pred:
            return False
        try:
            if _grade_answer_mathd(pred, gt):
                return True
            return _grade_answer_sympy(pred, gt)
        except ImportError:
            raise ModelError(
                "scoring='sympy' 需要 sympy（论文评分依赖）；请 pip install sympy")

    def evaluate(self, dataset_ref: str) -> AimeResult:
        """在单个 AIME 数据集上评估。

        - metric="pass1"（默认）：n_samples>1 记 pass@1（任一采样答对即对）。
        - metric="ave"（对齐论文 ave@32）：accuracy = 每题 n 采样中答对比例的均值。
        """
        problems = self.load_problems(dataset_ref)
        prompts = [format_prompt(p, self.prompt_style) for p, _ in problems]
        responses = self.generate(prompts)          # 拍平：N × n_samples 条
        n = self.n_samples
        correct = 0
        fracs: list[float] = []
        rows = []
        for i, (problem, gt) in enumerate(problems):
            group = responses[i * n:(i + 1) * n]
            if self.scoring == "sympy":
                # 论文协议：\boxed{} 级联提取 + 数学等价判定（grade_answer_mathd or sympy）
                preds = [(_extract_boxed_answer(r) or "") for r in group]
                per = [self._grade_sympy(p, gt) for p in preds]
            else:
                preds = [extract_answer(r, self.prompt_style) for r in group]
                gt_n = normalize_answer(gt)
                per = [normalize_answer(p) == gt_n for p in preds]
            ok = any(per)
            correct += int(ok)
            if self.metric == "ave" and n:
                fracs.append(sum(per) / n)
            rows.append({
                "problem_id": i, "dataset": dataset_ref,
                "ground_truth": gt, "predicted": preds[0],
                "correct": ok, "response": group[0],
                "n_samples": n,
                # 采样级审计数据（二次审阅修复）：每题 n 采样中【正确数】与全部预测答案——
                # 使 ave@32 可从落盘精确重算（用户指出原 jsonl 只存 pass@1 的 correct，
                # 无法复现 ave@32）。correct_count/n = 该题正确比例；30 题平均 = ave@32。
                "correct_count": int(sum(per)),
                "preds_all": preds,
            })
        ave = (sum(fracs) / len(fracs)) if fracs else None
        return AimeResult(dataset=dataset_ref, model_path=self.model_path,
                          correct=correct, total=len(problems), rows=rows,
                          ave_accuracy=ave)

    def evaluate_to_jsonl(self, dataset_ref: str, out_path: str) -> AimeResult:
        """评估并落盘每样本 jsonl，返回结果。"""
        res = self.evaluate(dataset_ref)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in res.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return res


__all__ = ["AimeEvaluator", "AimeResult", "extract_answer", "normalize_answer",
           "format_prompt", "AIME_DATASETS", "DEFAULT_DATASETS",
           "PROMPT_TEMPLATE", "PROMPT_TEMPLATE_DAPO", "_validate_device"]